enable wgpu_ray_query;

const PI: f32 = 3.14159265359;
const LAMBDA_MIN = 380.0;
const LAMBDA_MAX = 780.0;
const LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;


// ------------------------------------------
// 1. 構造体とバインディング
// ------------------------------------------
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

// --- 構造体とバインディングの追加 ---
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

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read> lights: array<LightInfo>;
@group(0) @binding(2) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(3) var<storage, read_write> photon_count: atomic<u32>;
@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<storage, read> materials: array<Material>;
@group(0) @binding(6) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(7) var<storage, read> indices: array<u32>;
@group(0) @binding(8) var<storage, read> mesh_infos: array<MeshInfo>;

// ------------------------------------------
// 2. 乱数生成とサンプリング
// ------------------------------------------
var<private> rng_seed: u32;

fn init_rng(linear_id: u32, frame: u32) {
    // スレッドIDとフレームカウントを掛け合わせて完全にばらけさせる
    rng_seed = linear_id + frame * 927163u;
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

// プランクの法則 (放射輝度計算)
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

// ------------------------------------------
// 3. メイン処理 (フォトン発射)
// ------------------------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let max_photons = 1024u * 1024u; 
    if id.x >= max_photons { return; }

    init_rng(id.x, camera.frame_count);

    let num_lights = arrayLength(&lights);
    if num_lights == 0u { return; }

    let light_idx = min(u32(rand() * f32(num_lights)), num_lights - 1u);
    let light = lights[light_idx];

// --- 起点と方向の計算 ---
    let edge1 = light.v1.xyz - light.v0.xyz;
    let edge2 = light.v2.xyz - light.v0.xyz;
    let normal = normalize(cross(edge1, edge2));
    let light_area = length(cross(edge1, edge2)) * 0.5;

    let r1 = rand(); let r2 = rand(); let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1; let v = r2 * sqrt_r1; let w = 1.0 - u - v;
    let emit_pos = light.v0.xyz * u + light.v1.xyz * v + light.v2.xyz * w;
    let emit_dir = normalize(normal + random_unit_vector());

    let temp_k = light.params[1];
    let intensity = light.params[2];

    // ==========================================
    // 視覚的重点サンプリング (Analytical Inverse Transform Sampling)
    // ==========================================
    let peak = 555.0;  // 視感度のピーク
    let gamma = 40.0;  // 分布の広がり (コーシー分布)

    // 1. CDF (累積分布関数) の最小・最大値を計算
    let cdf_min = (1.0 / PI) * atan((LAMBDA_MIN - peak) / gamma) + 0.5;
    let cdf_max = (1.0 / PI) * atan((LAMBDA_MAX - peak) / gamma) + 0.5;
    let cdf_range = cdf_max - cdf_min;

    // 2. 乱数 rnd_val を [cdf_min, cdf_max] にマッピング 
    let rnd_val = cdf_min + rand() * cdf_range;

    // 3. 解析的な逆関数を用いて O(1) で波長を決定！
    var wavelength = peak + gamma * tan(PI * (rnd_val - 0.5));
    wavelength = clamp(wavelength, LAMBDA_MIN, LAMBDA_MAX);

    // 4. 選択された波長の PDF (確率密度関数) を計算
    let pdf_norm = 1.0 / (cdf_range * PI * gamma);
    let diff_w = (wavelength - peak) / gamma;
    let pdf = pdf_norm / (1.0 + diff_w * diff_w);

    // 5. 物理的に正しい Flux の計算 (Radiance / PDF)
    let radiance = blackbody_radiance(wavelength, temp_k);
    let flux = (radiance / pdf) * light_area * PI;

    var current_energy = flux * intensity;

    // ==========================================
    // レイの追跡開始！
    // ==========================================
    var ray_pos = emit_pos + emit_dir * 0.001; // 自己交差回避
    var ray_dir = emit_dir;

    var is_caustic = false;

    // 最大5回バウンスさせる（ガラスを通って壁に落ちるには最低2〜3回必要）
    for (var depth = 0; depth < 5; depth++) {
        const T_MIN = 0.0001;
        const T_MAX = 100.0;
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u,0xFF, T_MIN, T_MAX, ray_pos, ray_dir));
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
        let hit_pos = ray_pos + ray_dir * hit.t;

        // --- 材質ごとの処理 ---
        if mat_type == 0u { 

            if is_caustic {
                let idx = atomicAdd(&photon_count, 1u);
                if idx < max_photons {
                    photons[idx].position = hit_pos;
                    // 後でカメラから集める時のために、光が入ってきた方向(-ray_dir)を記録しておく
                    photons[idx].direction = -ray_dir; 
                    photons[idx].wavelength = wavelength;
                    // 壁の反射率(RGB->XYZ近似など)を掛けるのが正確
                    photons[idx].energy = current_energy; 
                }
            }
            break; // 定着したので追跡終了

        } else if mat_type == 1u {
            // 金属(Metal) -> 反射してさらに飛ばす
            ray_dir = reflect(normalize(ray_dir), ffnormal);
            ray_pos = hit_pos + ffnormal * 0.001;
            
        } else if mat_type == 2u {
            // ガラス(Dielectric) -> カメラパスと同じく波長分散させて屈折！
            let A = mat.extra.z; let B = mat.extra.w;
            let lambda_um = wavelength / 1000.0;
            let ir = A + (B / (lambda_um * lambda_um));
            let refraction_ratio = select(ir, 1.0 / ir, is_front_face);
            let unit_dir = normalize(ray_dir);
            let cos_theta = min(dot(-unit_dir, ffnormal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            
            // フレネル(Schlickの近似)
            var r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
            r0 = r0 * r0;
            let reflectance = r0 + (1.0 - r0) * pow((1.0 - cos_theta), 5.0);

            if refraction_ratio * sin_theta > 1.0 || reflectance > rand() {
                ray_dir = reflect(unit_dir, ffnormal);
                let offset_normal = select(-ffnormal, ffnormal, dot(ray_dir, ffnormal) > 0.0);
                ray_pos = hit_pos + offset_normal * 0.001;
            } else {
                is_caustic = true;
                ray_dir = refract(unit_dir, ffnormal, refraction_ratio);
                
                let offset_normal = select(-ffnormal, ffnormal, dot(ray_dir, ffnormal) > 0.0);
                ray_pos = hit_pos + offset_normal * 0.001;
            }
        } else {
            break; // 光源などに当たったら終了
        }
    }
}