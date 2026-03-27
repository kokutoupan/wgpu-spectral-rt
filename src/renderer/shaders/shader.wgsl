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

struct LightSample {
    dir: vec3f,       // 光源への方向
    dist: f32,        // 光源までの距離
    pdf: f32,         // 立体角PDF
    radiance: vec4f,  // 輝度(スペクトル)
}

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

// ==========================================
// 薄膜干渉 (Thin-Film Interference)
// ==========================================
// n1: 外側の屈折率 (空気 = 1.0)
// n2: 膜の屈折率 (水/石鹸水 = 1.33)
// n3: 内側の屈折率 (シャボン玉なら空気=1.0、コーティングガラスなら1.5)
// d: 膜の厚さ (nm)
fn fresnel_thin_film(cos_theta1: f32, n1: f32, n2: f32, n3: f32, d: f32, wavelengths: vec4f) -> vec4f {
    // 1. スネルの法則で各層の角度(cos)を計算
    let sin_theta1_sq = max(0.0, 1.0 - cos_theta1 * cos_theta1);
    
    let sin_theta2_sq = (n1 / n2) * (n1 / n2) * sin_theta1_sq;
    if sin_theta2_sq >= 1.0 { return vec4f(1.0); } // 全反射
    let cos_theta2 = sqrt(1.0 - sin_theta2_sq);

    let sin_theta3_sq = (n1 / n3) * (n1 / n3) * sin_theta1_sq;
    if sin_theta3_sq >= 1.0 { return vec4f(1.0); } // 全反射
    let cos_theta3 = sqrt(1.0 - sin_theta3_sq);

    // 2. 境界面での振幅反射率 (s偏光・p偏光の平均を取る代わりに簡易的な垂直入射近似に近い形)
    let r12 = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2);
    let r23 = (n2 * cos_theta2 - n3 * cos_theta3) / (n2 * cos_theta2 + n3 * cos_theta3);

    let r12_sq = r12 * r12;
    let r23_sq = r23 * r23;

    // 3. 位相差 (Phase Difference) を波長(vec4f)ごとに計算！
    // 経路差による位相ズレ: δ = (4 * π * n2 * d * cos_theta2) / λ
    let delta = (4.0 * PI * n2 * d * cos_theta2) / wavelengths;
    let cos_delta = vec4f(cos(delta.x), cos(delta.y), cos(delta.z), cos(delta.w));

    // 4. Airyの公式 (Airy's Formula) による多重干渉を含んだ最終的な反射率 (vec4f)
    let num = r12_sq + r23_sq + 2.0 * r12 * r23 * cos_delta;
    let den = 1.0 + r12_sq * r23_sq + 2.0 * r12 * r23 * cos_delta;
    
    return clamp(num / den, vec4f(0.0), vec4f(1.0));
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

// グリッド・ジッター計算関数
fn get_grid_jitter(frame_count: u32, cell_size: f32) -> vec3f {
    // フレームカウント専用の軽量PCGハッシュ
    let state = (frame_count * 114514u) * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let seed = (word >> 22u) ^ word;

    // 0~255 のビットを抽出して 0.0 ~ 1.0 の乱数3つに変換
    let rx = f32(seed & 0xFFu) / 255.0;
    let ry = f32((seed >> 8u) & 0xFFu) / 255.0;
    let rz = f32((seed >> 16u) & 0xFFu) / 255.0;
    
    // セルサイズの範囲 [-0.5, 0.5] * cell_size でオフセットを返す
    return (vec3f(rx, ry, rz) - vec3f(0.5)) * cell_size;
}

fn gather_photons(hit_pos: vec3f, normal: vec3f, camera_wavelengths: vec4f) -> vec4f {
    var total_energy = vec4f(0.0);
    let jitter = get_grid_jitter(camera.frame_count, CELL_SIZE);
    let grid_pos = vec3i(floor((hit_pos + jitter) / CELL_SIZE));
    
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
    } else if mat_type== 4u{
        // 薄膜干渉 (Thin-Film Interference)
        let n1 = 1.0; // 外側 (空気)
        let n2 = mat.extra.z;
        let n3 = mat.extra.w;
        let d = max(mat.extra.y, 50.0);

        let unit_dir = normalize(r_dir);
        let cos_theta1 = min(dot(-unit_dir, ffnormal), 1.0);
        
        // 4波長分の干渉反射率を計算！
        let F_spectral = fresnel_thin_film(cos_theta1, n1, n2, n3, d, wavelengths);
        
        // Hero波長(x)の反射率を基準にして、反射か透過をロシアンルーレットで決定
        let reflect_prob = F_spectral.x;

        if rand() < reflect_prob {
            // 反射 (干渉による色がつく！)
            rec.scatter_dir = reflect(unit_dir, ffnormal);
            // 確率で割ってバイアスを消し、波長ごとの色を掛ける
            rec.attenuation = F_spectral / reflect_prob; 
        } else {
            // 透過 (シャボン玉の膜が薄すぎるため屈折による曲がりは無視して直進させる)
            // 透過光も干渉の影響を受けるため (1.0 - F) の色になる
            rec.scatter_dir = unit_dir;
            let T_spectral = vec4f(1.0) - F_spectral;
            rec.attenuation = T_spectral / (1.0 - reflect_prob);
        }
    }
    else { 
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

fn sample_random_light(hit_pos: vec3f, wavelengths: vec4f) -> LightSample {
    var ls: LightSample;
    let num_lights = arrayLength(&lights);
    if num_lights == 0u {
        ls.pdf = 0.0; return ls;
    }

    // 1. 光源をランダムに1つ選ぶ
    let light_idx = min(u32(rand() * f32(num_lights)), num_lights - 1u);
    let light = lights[light_idx];

    // 2. 三角形光源の表面をサンプリング
    let r1 = rand(); let r2 = rand(); let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1; let v = r2 * sqrt_r1; let w = 1.0 - u - v;
    let p = light.v0.xyz * u + light.v1.xyz * v + light.v2.xyz * w;

    let edge1 = light.v1.xyz - light.v0.xyz;
    let edge2 = light.v2.xyz - light.v0.xyz;
    let normal = normalize(cross(edge1, edge2));
    let area = length(cross(edge1, edge2)) * 0.5;

    // 3. 方向と距離の計算
    let dir_to_light = p - hit_pos;
    let dist_sq = dot(dir_to_light, dir_to_light);
    let dist = sqrt(dist_sq);
    let l = dir_to_light / dist;
    
    let cos_light = max(dot(normal, -l), 0.0);

    // 裏面から当たった場合は無効
    if cos_light <= 0.0001 {
        ls.pdf = 0.0; return ls;
    }

    ls.dir = l;
    ls.dist = dist;
    
    // 4. 面積PDFを立体角PDFに変換
    ls.pdf = dist_sq / (area * f32(num_lights) * cos_light);

    // 5. 光源のスペクトルエネルギーを計算
    let temp_k = light.params[1];
    let intensity = light.params[2];
    ls.radiance = vec4f(
        blackbody_radiance(wavelengths.x, temp_k),
        blackbody_radiance(wavelengths.y, temp_k),
        blackbody_radiance(wavelengths.z, temp_k),
        blackbody_radiance(wavelengths.w, temp_k)
    ) * intensity;

    return ls;
}

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

    var is_specular_bounce = true; // カメラレイやガラス・金属からの反射はtrue
    var prev_bsdf_pdf = 0.0;       // 前回のバウンスの確率密度

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
                var mis_weight = 1.0;
                // 直前のバウンスがDiffuse(NEEが行われた)だった場合のみウェイトを下げる
                if !is_specular_bounce {
                    // ヒットした光源ポリゴンの面積をその場で計算する！
                    let mesh_info = mesh_infos[mesh_id];
                    let idx_offset = mesh_info.index_offset + hit.primitive_index * 3u;
                    let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
                    let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
                    let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;
                    
                    let v0 = vertices[i0].pos.xyz;
                    let v1 = vertices[i1].pos.xyz;
                    let v2 = vertices[i2].pos.xyz;
                    
                    // 外積で三角形の面積を計算
                    let area = length(cross(v1 - v0, v2 - v0)) * 0.5;
                    
                    // 光源の立体角PDFを計算
                    let cos_light = max(dot(ffnormal, -r.dir), 0.0001);
                    let dist_sq = hit.t * hit.t;
                    let num_lights = f32(arrayLength(&lights));
                    
                    let light_pdf = dist_sq / (area * cos_light * num_lights);
                    
                    // Balance Heuristic
                    mis_weight = prev_bsdf_pdf / (prev_bsdf_pdf + light_pdf);
                }
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
            let hit_pos = r.origin + r.dir * hit.t;

            // 1. Photon Mapping (ガラスの集光 / コースティクス)
            let gathered_light = gather_photons(hit_pos, ffnormal, wavelengths);
            // // 壁の色(スペクトル)と掛け合わせて、最終的な色とする
            accumulated_color += throughput *( mat_spectral_color/PI) * gathered_light;

            // 2. Light Sampling (直接光)
            let ls = sample_random_light(hit_pos, wavelengths);
            if ls.pdf > 0.0 {
                // 光源に向かってシャドウレイを飛ばす！
                var shadow_rq: ray_query;
                rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0xFFu, 0.001, ls.dist - 0.001, hit_pos, ls.dir));
                rayQueryProceed(&shadow_rq);
                
                // 何も遮るものがなければ(光源が見えたら)加算
                if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {
                    let cos_theta = max(dot(ffnormal, ls.dir), 0.0);
                    
                    let bsdf_eval = mat_spectral_color / PI; 
                    let bsdf_pdf = cos_theta / PI; // ランバートのコサインサンプリングのPDF

                    // 【MISウェイトの計算 (Balance Heuristic)】
                    let mis_weight = ls.pdf / (ls.pdf + bsdf_pdf);

                    // 寄与の加算: Throughput * BSDF * Radiance * cos_theta / PDF * MIS_Weight
                    accumulated_color += throughput * bsdf_eval * ls.radiance * cos_theta * (mis_weight / ls.pdf);
                }
            }

        }


        // --- BSDF ---
        let scatter_rec = sample_bsdf(r.dir, ffnormal, mat, is_front_face, wavelengths,throughput);
        if scatter_rec.absorbed { break; }

        if mat_type == 0u {
            is_specular_bounce = false; // DiffuseなのでNEEが発動した
            // DiffuseのコサインサンプリングのPDF
            let cos_theta = max(dot(ffnormal, scatter_rec.scatter_dir), 0.0001);
            prev_bsdf_pdf = cos_theta / PI;
        } else {
            // ガラスや金属はNEEをしていないので、次に光源に当たったらウェイトは1.0のままにする
            is_specular_bounce = true; 
            prev_bsdf_pdf = 0.0;
        }
        
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
    // Hero Wavelength Sampling (Analytical Inverse Transform Sampling)
    // ------------------------------------------

    let peak = 555.0;  // 人間の目の感度ピーク (緑〜黄)
    let gamma = 40.0;  // 分布の広がり具合

    // 1. CDF (累積分布関数) の最小・最大値を計算
    // CDF(x) = (1/PI) * atan((x - peak) / gamma) + 0.5
    let cdf_min = (1.0 / PI) * atan((LAMBDA_MIN - peak) / gamma) + 0.5;
    let cdf_max = (1.0 / PI) * atan((LAMBDA_MAX - peak) / gamma) + 0.5;
    let cdf_range = cdf_max - cdf_min;

    // 2. 乱数 u (0.0~1.0) を [cdf_min, cdf_max] にマッピング
    let u = cdf_min + rand() * cdf_range;

    // x = peak + gamma * tan(PI * (u - 0.5))
    var hero_lambda = peak + gamma * tan(PI * (u - 0.5));
    hero_lambda = clamp(hero_lambda, LAMBDA_MIN, LAMBDA_MAX);

    // 4波長を展開
    var wavelengths = vec4f(
        hero_lambda,
        hero_lambda + (LAMBDA_RANGE / 4.0) * 1.0,
        hero_lambda + (LAMBDA_RANGE / 4.0) * 2.0,
        hero_lambda + (LAMBDA_RANGE / 4.0) * 3.0
    );
    wavelengths = LAMBDA_MIN + (wavelengths - LAMBDA_MIN) % LAMBDA_RANGE;

    let rot = rand();
    if rot < 0.25 {
    } else if rot < 0.5 {
        wavelengths = wavelengths.yzwx;
    } else if rot < 0.75 {
        wavelengths = wavelengths.zwxy;
    } else {
        wavelengths = wavelengths.wxyz;
    }

    let spectral_radiance = ray_color(ray, wavelengths);

    // ------------------------------------------
    // Spectral to RGB & MIS Weight
    // ------------------------------------------
    var xyz = vec3f(0.0);
    xyz += sample_cie_xyz(wavelengths.x) * spectral_radiance.x;
    xyz += sample_cie_xyz(wavelengths.y) * spectral_radiance.y;
    xyz += sample_cie_xyz(wavelengths.z) * spectral_radiance.z;
    xyz += sample_cie_xyz(wavelengths.w) * spectral_radiance.w;

    // 4. 数学的に完全に正しい PDF (確率密度関数) の計算
    // PDF(x) = 1 / (cdf_range * PI * gamma * (1 + ((x - peak)/gamma)^2))
    let pdf_norm = 1.0 / (cdf_range * PI * gamma);
    
    let diff0 = (wavelengths.x - peak) / gamma;
    let p0 = pdf_norm / (1.0 + diff0 * diff0);
    
    let diff1 = (wavelengths.y - peak) / gamma;
    let p1 = pdf_norm / (1.0 + diff1 * diff1);
    
    let diff2 = (wavelengths.z - peak) / gamma;
    let p2 = pdf_norm / (1.0 + diff2 * diff2);
    
    let diff3 = (wavelengths.w - peak) / gamma;
    let p3 = pdf_norm / (1.0 + diff3 * diff3);

    let sum_pdf = p0 + p1 + p2 + p3;
    xyz *= 1.0 / sum_pdf;

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