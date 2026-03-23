enable wgpu_ray_query;

// ==========================================
// 1. Constants & Structs
// ==========================================
const PI: f32 = 3.14159265359;
override MAX_DEPTH: u32 = 8u;

struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    view: mat4x4f,
    proj: mat4x4f,
    frame_count: u32,
}

struct Material {
    color: vec4f,
    extra: vec4f, 
}

struct Vertex {
    pos: vec4f,
    normal: vec4f,
}

struct MeshInfo {
    vertex_offset: u32,
    index_offset: u32,
    pad: vec2u,
}

struct LightInfo {
    v0: vec4f,
    v1: vec4f,
    v2: vec4f,
    params: vec4f,
}

struct Photon {
    position: vec3f,
    wavelength: f32,
    direction: vec3f,
    energy: f32,
}

struct Ray {
    origin: vec3f,
    dir: vec3f,
}

struct ScatterRecord {
    scatter_dir: vec3f,
    attenuation: vec4f, 
    absorbed: bool,
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read_write> accumulation: array<vec4f>;
@group(0) @binding(5) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(6) var<storage, read> indices: array<u32>;
@group(0) @binding(7) var<storage, read> mesh_infos: array<MeshInfo>;
@group(0) @binding(8) var<storage, read> lights: array<LightInfo>;
@group(0) @binding(9) var<storage, read> photons: array<Photon>;
@group(0) @binding(10) var<storage, read> grid_head: array<u32>;
@group(0) @binding(11) var<storage, read> grid_next: array<u32>;

// ==========================================
// 1.5 Spectral & Color Math
// ==========================================
const LAMBDA_MIN = 380.0;
const LAMBDA_MAX = 780.0;
const LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;

fn sample_cie_xyz(lambda: f32) -> vec3f {
    let x = exp(-pow((lambda - 599.8) / 37.9, 2.0)) * 1.056
          + exp(-pow((lambda - 442.0) / 16.0, 2.0)) * 0.362
          - exp(-pow((lambda - 501.1) / 26.7, 2.0)) * 0.065;
          
    let y = exp(-pow((lambda - 568.8) / 46.9, 2.0)) * 0.821
          + exp(-pow((lambda - 530.9) / 16.3, 2.0)) * 0.286;
          
    let z = exp(-pow((lambda - 437.0) / 11.8, 2.0)) * 1.217
          + exp(-pow((lambda - 459.0) / 26.0, 2.0)) * 0.681;
          
    return vec3f(max(0.0, x), max(0.0, y), max(0.0, z));
}

// WGSLはColumn-majorなので、縦の列(Col)ごとにvec3fを定義する必要があります！
const XYZ_TO_SRGB = mat3x3f(
    vec3f( 3.2404542, -0.9692660,  0.0556434), // Col 0 (Xに対する R, G, B のウェイト)
    vec3f(-1.5371385,  1.8760108, -0.2040259), // Col 1 (Yに対する R, G, B のウェイト)
    vec3f(-0.4985314,  0.0415560,  1.0572252)  // Col 2 (Zに対する R, G, B のウェイト)
);
// [新規] RGBからスペクトルへのガウシアン変換
fn rgb_to_spectrum_eval(rgb: vec3f, lambda: f32) -> f32 {
    let white = min(rgb.r, min(rgb.g, rgb.b)); 
    let c = max(rgb - vec3f(white), vec3f(0.0)); 
    let gauss_width = 20.0;

    // R, G, B それぞれのピーク波長にガウス分布を配置
    let r_val = c.r * exp(-pow((lambda - 630.0) / gauss_width, 2.0));
    let g_val = c.g * exp(-pow((lambda - 525.0) / gauss_width, 2.0));
    let b_val = c.b * exp(-pow((lambda - 460.0) / gauss_width, 2.0));

return clamp(white + r_val + g_val + b_val, 0.0, 0.99);
}

fn rgb_to_spectrum_vec4(rgb: vec3f, wavelengths: vec4f) -> vec4f {
    return vec4f(
        rgb_to_spectrum_eval(rgb, wavelengths.x),
        rgb_to_spectrum_eval(rgb, wavelengths.y),
        rgb_to_spectrum_eval(rgb, wavelengths.z),
        rgb_to_spectrum_eval(rgb, wavelengths.w)
    );
}

fn blackbody_radiance(lambda_nm: f32, temp_k: f32) -> f32 {
    let lambda_m = lambda_nm * 1e-9;
    let h = 6.62607015e-34; 
    let c = 299792458.0;    
    let k = 1.380649e-23;   

    let c1 = 2.0 * h * c * c;
    let c2 = (h * c) / (k * temp_k);
    
    let radiance = c1 / (pow(lambda_m, 5.0) * (exp(c2 / lambda_m) - 1.0));
    return radiance * 1e-13; 
}

// ==========================================
// 2. Math & RNG
// ==========================================
var<private> rng_seed: u32;

fn init_rng(pos: vec2u, width: u32, frame: u32) {
    rng_seed = pos.x + pos.y * width + frame * 927163u;
    rng_seed = pcg_hash(rng_seed);
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_seed = pcg_hash(rng_seed);
    return f32(rng_seed) / 4294967295.0;
}

fn random_unit_vector() -> vec3f {
    let u1 = rand();
    let u2 = rand();
    let z = 1.0 - 2.0 * u1;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 2.0 * PI * u2;
    return vec3f(r * cos(phi), r * sin(phi), z);
}

fn build_orthonormal_basis(n: vec3f) -> mat3x3f {
    var up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let right = normalize(cross(up, n));
    up = cross(n, right);
    return mat3x3f(right, up, n);
}

// ==========================================
// 3. Microfacet BRDF (GGX) Functions
// ==========================================
fn sample_ggx_half_vector(n: vec3f, roughness: f32) -> vec3f {
    let a2 = roughness * roughness;
    let r1 = rand();
    let r2 = rand();

    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    let h_tangent = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    let tbn = build_orthonormal_basis(n);
    return normalize(tbn * h_tangent);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0; 
    let num = n_dot_v;
    let denom = n_dot_v * (1.0 - k) + k;
    return num / denom;
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

//  波長(vec4f)ごとのフレネル計算
fn fresnel_schlick_vec4(cos_theta: f32, f0: vec4f) -> vec4f {
    return f0 + (vec4f(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// ====================
// 3.5 Photon Mapping
// ====================

const HASH_SIZE: u32 = 4 * 1024 * 1024; 
const CELL_SIZE: f32 = 0.02;     
const GATHER_RADIUS: f32 = 0.02; // 検索半径(5cm)
const GATHER_RADIUS_SQ: f32 = GATHER_RADIUS * GATHER_RADIUS;
const MAX_PHOTON = 1048576u;
const SPECTRAL_RADIUS = 20.0;

fn gather_photons(hit_pos: vec3f, normal: vec3f, camera_wavelengths: vec4f) -> vec4f {
    var total_energy = vec4f(0.0);
    let grid_pos = vec3i(floor(hit_pos / CELL_SIZE));
    
    let initial_radius = 0.02;
    let current_radius = initial_radius * pow(f32(camera.frame_count + 1u), -0.15);
    let current_radius_sq = current_radius * current_radius;

    // ハッシュ計算 (build_grid.wgsl と完全に同じ式)
    let p1 = 73856093u; let p2 = 19349663u; let p3 = 83492791u;
    let h = (u32(grid_pos.x) * p1 ^ u32(grid_pos.y) * p2 ^ u32(grid_pos.z) * p3) & (HASH_SIZE - 1u);

    // 連結リストを辿る
    var photon_id = grid_head[h];
    for(var loop_count = 0; loop_count < 64; loop_count++) {
        if(photon_id == 0xFFFFFFFFu) {
            break;
        }
        let photon = photons[photon_id];
        let dist_sq = dot(hit_pos - photon.position, hit_pos - photon.position); // 距離の2乗
        
        if dist_sq < current_radius_sq {
            // 壁の裏側など、法線と逆向きのフォトンは除外
            if dot(normal, photon.direction) > 0.0 {
                let dist = sqrt(dist_sq);
                let spatial_weight = 1.0 - (dist / current_radius);

                let lambda_diff = abs(camera_wavelengths - vec4f(photon.wavelength));
    
                // 差が0なら重み1.0、20nm離れていたら重み0.0になる (ガウス分布の近似)
                let spectral_weight = max(vec4f(1.0) - (lambda_diff / SPECTRAL_RADIUS), vec4f(0.0));
                
                total_energy += photon.energy * spatial_weight * spectral_weight;
            }
        }
        photon_id = grid_next[photon_id];
    }
    
    // 密度推定：集めたエネルギーを「円の面積」と「発射した全フォトン数」で割る
    let spatial_norm  = 3.0 / (PI * current_radius_sq);
    let spectral_norm = 1.0 / SPECTRAL_RADIUS;
    let count_norm    = 1.0 / f32(MAX_PHOTON);
    let density_factor = spatial_norm * spectral_norm * count_norm;
    return total_energy * density_factor;
}

// ==========================================
// 4. BSDF Evaluation
// ==========================================
fn sample_bsdf(r_dir: vec3f, ffnormal: vec3f, mat: Material, is_front_face: bool, wavelengths: vec4f, throughput: vec4f) -> ScatterRecord {
    var rec: ScatterRecord;
    rec.absorbed = false;
    let mat_type = u32(mat.extra.x);
    let v = normalize(-r_dir); 
    
    // [変更] RGBを4波長のスペクトル分布に変換
    let mat_spectral_color = rgb_to_spectrum_vec4(mat.color.rgb, wavelengths);
    
    if mat_type == 1u { 
        let roughness = max(mat.extra.y, 0.001); 
        let h = sample_ggx_half_vector(ffnormal, roughness);
        let l = reflect(-v, h); 
        
        let n_dot_l = dot(ffnormal, l);
        let n_dot_v = dot(ffnormal, v);
        
        if n_dot_l > 0.0 && n_dot_v > 0.0 {
            let n_dot_h = max(dot(ffnormal, h), 0.0);
            let v_dot_h = max(dot(v, h), 0.0);
            
            // f0とフレネルは波長(vec4f)ごとに計算される
            let f0 = mat_spectral_color; 
            let F = fresnel_schlick_vec4(v_dot_h, f0);
            let G = geometry_smith(n_dot_v, n_dot_l, roughness);
            
            let weight = (F * G * v_dot_h) / max(n_dot_v * n_dot_h, 0.0001);
            
            rec.scatter_dir = l;
            rec.attenuation = weight; 
        } else {
            rec.absorbed = true;
        }
    } else if mat_type == 2u { 
// ------------------------------------------
        // Dielectric (Glass) with Dispersion
        // ------------------------------------------
        let A = mat.extra.z; // ベースの屈折率 (例: 1.5)
        let B = mat.extra.w; // 分散係数 (例: 0.005)
        
        // 主波長(Hero Wavelength: wavelengths.x)を um 単位に変換して屈折率を計算
        let lambda_um = wavelengths.x / 1000.0;
        let ir = A + (B / (lambda_um * lambda_um));
        
        let refraction_ratio = select(ir, 1.0 / ir, is_front_face);
        let unit_dir = normalize(r_dir);
        let cos_theta = min(dot(-unit_dir, ffnormal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        
        if refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > rand() {
            // 全反射(Total Internal Reflection) または フレネル反射 の場合
            // 反射は波長によって角度が変わらないので、4波長ともそのまま生き残る
            rec.scatter_dir = reflect(unit_dir, ffnormal);
            rec.attenuation = mat_spectral_color;
        } else {
            // 屈折(Refraction) の場合
            rec.scatter_dir = refract(unit_dir, ffnormal, refraction_ratio);
            
            // 【重要】波長ごとに屈折角が変わるため、1本のレイで追跡できるのは主波長(x)だけ！
            // 他の波長(y, z, w)のエネルギーをゼロにし、エネルギー保存のために残った x を4倍する
            if B > 0.0 {
                let is_collapsed = (throughput.y < 0.0001 && throughput.z < 0.0001 && throughput.w < 0.0001);
                
                // 初めての分散なら4倍、すでに分散済みなら1倍
                let multiplier = select(4.0, 1.0, is_collapsed);
                
                rec.attenuation = vec4f(mat_spectral_color.x * multiplier, 0.0, 0.0, 0.0);
            } else {
                rec.attenuation = mat_spectral_color; // 分散なしの場合はそのまま
            }
        }
    } else { 
        rec.scatter_dir = ffnormal + random_unit_vector();
        if length(rec.scatter_dir) < 0.001 {
            rec.scatter_dir = ffnormal;
        }
        rec.scatter_dir = normalize(rec.scatter_dir);
        rec.attenuation = mat_spectral_color;
    }
    
    return rec;
}

// ==========================================
// 5. Ray Tracing Core
// ==========================================
fn get_interpolated_normal(mesh_id: u32, primitive_index: u32, barycentric: vec2f) -> vec3f {
    let mesh_info = mesh_infos[mesh_id];
    let idx_offset = mesh_info.index_offset + primitive_index * 3u;
    let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
    let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
    let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;

    let n0 = vertices[i0].normal.xyz;
    let n1 = vertices[i1].normal.xyz;
    let n2 = vertices[i2].normal.xyz;

    let u = barycentric.x;
    let v = barycentric.y;
    let w = 1.0 - u - v;
    return normalize(n0 * w + n1 * u + n2 * v);
}

fn ray_color(r_in: Ray, wavelengths: vec4f) -> vec4f {
    const T_MIN = 0.0001;
    const T_MAX = 100.0;
    var r = r_in;
    var accumulated_color = vec4f(0.0);
    var throughput = vec4f(1.0);

    var has_hit_diffuse = false;
    var is_caustic_path = false;

    for (var i = 0u; i < MAX_DEPTH; i++) {
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, T_MIN, T_MAX, r.origin, r.dir));
        rayQueryProceed(&rq);

        let hit = rayQueryGetCommittedIntersection(&rq);
        if hit.kind == 0u { break; } 

        let raw_id = hit.instance_custom_data;
        let mesh_id = raw_id >> 16u;
        let mat_id = raw_id & 0xFFFFu;
        let mat = materials[mat_id];

        var local_normal = get_interpolated_normal(mesh_id, hit.primitive_index, hit.barycentrics);
        let w2o = hit.world_to_object;
        let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
        let world_normal = normalize(local_normal * m_inv);

        let is_front_face = hit.front_face;
        let ffnormal = select(-world_normal, world_normal, is_front_face);
        let mat_type = u32(mat.extra.x);
        
        // --- Emission ---
        // [変更] 発光もスペクトル変換を通す
        var mat_spectral_emission = vec4f(0.0);

        if mat_type == 3u {
            if is_caustic_path { break; }
            let temp_k = mat.color.r;
            let intensity = mat.color.g;
            
            // 4波長それぞれの黒体放射を計算
            mat_spectral_emission = vec4f(
                blackbody_radiance(wavelengths.x, temp_k),
                blackbody_radiance(wavelengths.y, temp_k),
                blackbody_radiance(wavelengths.z, temp_k),
                blackbody_radiance(wavelengths.w, temp_k)
            ) * intensity;
            
            let no_cull = mat.extra.y > 0.0;
            if is_front_face || no_cull {
                accumulated_color += mat_spectral_emission * throughput;
            }
            break; 
        }

        if mat_type == 0u {
            has_hit_diffuse = true;
            is_caustic_path = false;
        } else if mat_type == 2u {
            if has_hit_diffuse { is_caustic_path = true; }
        }

        // --- Photon Mapping (Diffuse Surfaces Only) ---
        if mat_type == 0u { // Diffuse
            let mat_spectral_color = rgb_to_spectrum_vec4(mat.color.rgb, wavelengths);
            // // 周辺のフォトンをかき集めて明るさを計算
            let hit_pos = r.origin + r.dir * hit.t;
            let gathered_light = gather_photons(hit_pos, ffnormal, wavelengths);
            // let gathered_light = vec4f(0.0);
            
            // // 壁の色(スペクトル)と掛け合わせて、最終的な色とする
            accumulated_color += throughput *( mat_spectral_color/PI) * gathered_light;
        }


        // --- BSDF ---
        let scatter_rec = sample_bsdf(r.dir, ffnormal, mat, is_front_face, wavelengths,throughput);
        if scatter_rec.absorbed { break; }
        
        throughput *= scatter_rec.attenuation;
        r.origin = (r.origin + r.dir * hit.t);
        r.dir = scatter_rec.scatter_dir;

        // --- Russian Roulette ---
        if (i > 3) {
            let p_survive = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
            if (rand() > p_survive) { break; }
            throughput /= p_survive;
        }
    }
    return accumulated_color;
}

// ==========================================
// 6. Compute Entry Point
// ==========================================
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_tex);
    if id.x >= size.x || id.y >= size.y { return; }

    init_rng(id.xy, size.x, camera.frame_count);

    let uv = vec2f(id.xy) / vec2f(size);
    let view_inv = mat4x4f(camera.view_inverse[0], camera.view_inverse[1], camera.view_inverse[2], camera.view_inverse[3]);
    let proj_inv = mat4x4f(camera.proj_inverse[0], camera.proj_inverse[1], camera.proj_inverse[2], camera.proj_inverse[3]);
    let origin = view_inv[3].xyz;

    let jitter = vec2f(rand(), rand());
    let uv_jittered = (vec2f(id.xy) + jitter) / vec2f(size);
    let ndc_jittered = vec2f(uv_jittered.x * 2.0 - 1.0, uv_jittered.y * 2.0 - 1.0);

    let target_ndc_jittered = vec4f(ndc_jittered, 1.0, 1.0);
    let target_world_jittered = view_inv * proj_inv * target_ndc_jittered;
    let direction_jittered = normalize(target_world_jittered.xyz / target_world_jittered.w - origin);

    let ray = Ray(origin, direction_jittered);

    // ------------------------------------------
    // Hero Wavelength Sampling
    // ------------------------------------------
    let hero_lambda = LAMBDA_MIN + rand() * LAMBDA_RANGE;
    var wavelengths = vec4f(
        hero_lambda,
        hero_lambda + (LAMBDA_RANGE / 4.0) * 1.0,
        hero_lambda + (LAMBDA_RANGE / 4.0) * 2.0,
        hero_lambda + (LAMBDA_RANGE / 4.0) * 3.0
    );
    wavelengths = LAMBDA_MIN + (wavelengths - LAMBDA_MIN) % LAMBDA_RANGE;

    let spectral_radiance = ray_color(ray, wavelengths);

    // ------------------------------------------
    // Spectral to RGB (XYZ Color Matching)
    // ------------------------------------------
    var xyz = vec3f(0.0);
    xyz += sample_cie_xyz(wavelengths.x) * spectral_radiance.x;
    xyz += sample_cie_xyz(wavelengths.y) * spectral_radiance.y;
    xyz += sample_cie_xyz(wavelengths.z) * spectral_radiance.z;
    xyz += sample_cie_xyz(wavelengths.w) * spectral_radiance.w;
    
    xyz *= LAMBDA_RANGE / 4.0;

    // Linear sRGB に変換し、トーンマッピング(ACES近似など)を挟まずにそのまま蓄積
    let pixel_color_linear = XYZ_TO_SRGB * xyz;

    let idx = id.y * size.x + id.x;
    var current_acc = vec4f(0.0);
    if camera.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(pixel_color_linear, 1.0);
    accumulation[idx] = new_acc;
    let exposure = 1.0;
    // 蓄積結果の平均化
    var  final_color = max(new_acc.rgb / new_acc.w * exposure, vec3f(0.0));
    // 1. 人間の目が感じる「明るさ(輝度: Luma)」を計算
    let luma = dot(final_color, vec3f(0.2126, 0.7152, 0.0722));
    
    // 2. 「明るさ」だけを 0.0 ~ 1.0 に圧縮 (Reinhard)
    let mapped_luma = luma / (luma + 1.0);
    
    // 3. 元の色の比率を保ったまま、圧縮した明るさを適用する
    if luma > 0.00001 {
        final_color = final_color * (mapped_luma / luma);
    }
    textureStore(out_tex, id.xy, vec4f(final_color, 1.0));
}