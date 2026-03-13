@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<uniform> scale_offset: vec4f; // xy: scale, zw: offset

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    // 2枚の三角形を作る (0,1,2, 0,2,3 みたいな)
    var uvs = array<vec2f, 6>(
        vec2f(0.0, 0.0), // Top Left
        vec2f(1.0, 0.0), // Top Right
        vec2f(1.0, 1.0), // Bottom Right

        vec2f(0.0, 0.0), // Top Left
        vec2f(1.0, 1.0), // Bottom Right
        vec2f(0.0, 1.0)  // Bottom Left
    );

    let uv = uvs[idx % 6u];
    
    // Convert 0..1 UV to -1..1 NDC
    let ndc = vec2f(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0); 

    // Apply scale and offset so the quad fits correctly inside the screen space
    let scaled_ndc = ndc * scale_offset.xy + scale_offset.zw;

    return VSOut(vec4f(scaled_ndc, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    return textureSample(t, s, in.uv);
}
