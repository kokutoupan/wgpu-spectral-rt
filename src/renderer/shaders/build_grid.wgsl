struct Photon {
    position: vec3f,
    wavelength: f32,
    direction: vec3f,
    energy: f32,
}

struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    view: mat4x4f,
    proj: mat4x4f,
    frame_count: u32,
}

@group(0) @binding(0) var<storage, read> photons: array<Photon>;
@group(0) @binding(1) var<storage, read> photon_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> grid_head: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_next: array<u32>;
@group(0) @binding(4) var<uniform> camera: Camera;

const HASH_SIZE: u32 = 4 *1048576u; // 1024 * 1024
const CELL_SIZE: f32 = 0.02;     // 1マスを2cmとする

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


fn hash_position(p: vec3f) -> u32 {
    let jitter = get_grid_jitter(camera.frame_count, CELL_SIZE);
    let grid_pos = vec3i(floor((p + jitter) / CELL_SIZE));
    
    // 空間座標を巨大な素数でバラバラにハッシュ化
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    
    let h = u32(grid_pos.x) * p1 ^ u32(grid_pos.y) * p2 ^ u32(grid_pos.z) * p3;
    return h & (HASH_SIZE - 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let count = atomicLoad(&photon_count);
    if id.x >= count { return; }

    let photon = photons[id.x];
    
    // エネルギーがない（空中に飛んでいった）フォトンは無視
    if photon.energy == 0.0 {
        return;
    }

    // 1. 自分がいる空間のハッシュ値(マス目)を計算
    let h = hash_position(photon.position);

    // 2. そのマス目の先頭(head)に自分を書き込み、直前に書かれていたIDを貰う！
    let prev_id = atomicExchange(&grid_head[h], id.x);

    // 3. 自分と同じマス目にいた「前のフォトン」のIDを、自分の next に記録する
    grid_next[id.x] = prev_id;
}