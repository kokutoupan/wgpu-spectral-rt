enable wgpu_ray_query;

// ==========================================
// 1. Constants & Structs
// ==========================================
const PI: f32 = 3.14159265359;
override MAX_DEPTH: u32 = 8u;

struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    frame_count: u32,
}

struct Material {
    color: vec4f,
    emission: vec4f,
    extra: vec4f, // x: type, y: fuzz/roughness, z: ior, w: padding
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

struct Ray {
    origin: vec3f,
    dir: vec3f,
}

struct ScatterRecord {
    scatter_dir: vec3f,
    attenuation: vec3f,
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

// 任意の法線を基準とした正規直交基底(TBN)マトリクスを作る
fn build_orthonormal_basis(n: vec3f) -> mat3x3f {
    var up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let right = normalize(cross(up, n));
    up = cross(n, right);
    return mat3x3f(right, up, n);
}

// ==========================================
// 3. Microfacet BRDF (GGX) Functions
// ==========================================

// ラフネスからハーフベクトル(H)を重点サンプリングする
fn sample_ggx_half_vector(n: vec3f, roughness: f32) -> vec3f {
    let a2 = roughness * roughness;
    let r1 = rand();
    let r2 = rand();

    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    // 接空間でのベクトル
    let h_tangent = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    // ワールド空間へ変換
    let tbn = build_orthonormal_basis(n);
    return normalize(tbn * h_tangent);
}

// Schlick-GGX 幾何減衰 (パストレーシング用 k = a^2 / 2)
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

fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// ==========================================
// 4. BSDF Evaluation
// ==========================================
fn sample_bsdf(r_dir: vec3f, ffnormal: vec3f, mat: Material, is_front_face: bool) -> ScatterRecord {
    var rec: ScatterRecord;
    rec.absorbed = false;
    let mat_type = u32(mat.extra.x);

    let v = normalize(-r_dir); // 視線方向 (カメラまたは直前のバウンスから自分に向かってくるベクトル)
    
    if mat_type == 1u { 
        // ------------------------------------------
        // GGX Microfacet (Metal)
        // ------------------------------------------
        // roughnessが0だと計算が破綻するので最小値を設ける
        let roughness = max(mat.extra.y, 0.001); 
        
        // 1. ハーフベクトル H をサンプリング
        let h = sample_ggx_half_vector(ffnormal, roughness);
        
        // 2. 入射方向 L (scatter_dir) を決定
        let l = reflect(-v, h); 
        
        let n_dot_l = dot(ffnormal, l);
        let n_dot_v = dot(ffnormal, v);
        
        if n_dot_l > 0.0 && n_dot_v > 0.0 {
            let n_dot_h = max(dot(ffnormal, h), 0.0);
            let v_dot_h = max(dot(v, h), 0.0);
            
            // 3. フレネルと幾何減衰の計算
            let f0 = mat.color.rgb; // 金属の場合、ベースカラーがそのまま反射率(F0)になる
            let F = fresnel_schlick(v_dot_h, f0);
            let G = geometry_smith(n_dot_v, n_dot_l, roughness);
            
            // 4. Importance Sampling のウェイト計算
            // 数学的な期待値: (F * G * v_dot_h) / (n_dot_v * n_dot_h)
            let weight = (F * G * v_dot_h) / max(n_dot_v * n_dot_h, 0.0001);
            
            rec.scatter_dir = l;
            rec.attenuation = weight;
        } else {
            rec.absorbed = true;
        }
        
    } else if mat_type == 2u { 
        // ------------------------------------------
        // Dielectric (Glass)
        // ------------------------------------------
        let ir = mat.extra.z;
        let refraction_ratio = select(ir, 1.0 / ir, is_front_face);
        let unit_dir = normalize(r_dir);
        let cos_theta = min(dot(-unit_dir, ffnormal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        
        if refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > rand() {
            rec.scatter_dir = reflect(unit_dir, ffnormal);
        } else {
            rec.scatter_dir = refract(unit_dir, ffnormal, refraction_ratio);
        }
        rec.attenuation = mat.color.rgb;
        
    } else { 
        // ------------------------------------------
        // Lambertian (Diffuse)
        // ------------------------------------------
        rec.scatter_dir = ffnormal + random_unit_vector();
        if length(rec.scatter_dir) < 0.001 {
            rec.scatter_dir = ffnormal;
        }
        rec.scatter_dir = normalize(rec.scatter_dir);
        rec.attenuation = mat.color.rgb;
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

fn ray_color(r_in: Ray) -> vec3f {
    const T_MIN = 0.0001;
    const T_MAX = 100.0;
    var r = r_in;
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);

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
        if mat_type == 3u {
            let no_cull = mat.extra.y > 0.0;
            if is_front_face || no_cull {
                accumulated_color += mat.emission.rgb * throughput;
            }
            break; 
        }
        accumulated_color += mat.emission.rgb * throughput;

        // --- BSDF ---
        let scatter_rec = sample_bsdf(r.dir, ffnormal, mat, is_front_face);
        if scatter_rec.absorbed {
            break;
        }
        throughput *= scatter_rec.attenuation;
        r.origin = r.origin + r.dir * hit.t;
        r.dir = scatter_rec.scatter_dir;

        // --- Russian Roulette ---
        if (i > 3) {
            let p_survive = max(throughput.x, max(throughput.y, throughput.z));
            if (rand() > p_survive) {
                break;
            }
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
    let pixel_color_linear = ray_color(ray);

    let idx = id.y * size.x + id.x;
    var current_acc = vec4f(0.0);
    if camera.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(pixel_color_linear, 1.0);
    accumulation[idx] = new_acc;

    let final_color = new_acc.rgb / new_acc.w;
    textureStore(out_tex, id.xy, vec4f(final_color, 1.0));
}