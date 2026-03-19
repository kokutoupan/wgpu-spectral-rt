struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    view: mat4x4f,
    proj: mat4x4f,
    frame_count: u32,
}

struct Photon {
    position: vec3f,
    pad1: u32,
    direction: vec3f,
    pad2: u32,
    wavelengths: vec4f,
    energy: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> photons: array<Photon>;

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    var out: VertexOutput;
    let photon = photons[id];

    // まだ書き込まれていないフォトン(エネルギー0)は画面外へ飛ばして無視
    if (photon.energy.x == 0.0 && photon.energy.y == 0.0) {
        out.clip_position = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    // 3Dのワールド座標を、2Dの画面座標(NDC)に変換
    let view_pos = camera.view * vec4f(photon.position, 1.0);
    out.clip_position = camera.proj * view_pos;

    // 半透明のオレンジ色（重なると加算合成で光る）
    out.color = vec4f(1.0, 0.6, 0.1, 0.3); 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}