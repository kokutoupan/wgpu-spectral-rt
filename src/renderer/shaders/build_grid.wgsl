struct Photon {
    position: vec3f,
    wavelength: f32,
    direction: vec3f,
    energy: f32,
}

@group(0) @binding(0) var<storage, read> photons: array<Photon>;
@group(0) @binding(1) var<storage, read> photon_count: atomic<u32>;
// 各マス目の「先頭のフォトンID」
@group(0) @binding(2) var<storage, read_write> grid_head: array<atomic<u32>>;
// 各フォトンの「次のフォトンID」
@group(0) @binding(3) var<storage, read_write> grid_next: array<u32>;

const HASH_SIZE: u32 = 4 *1048576u; // 1024 * 1024
const CELL_SIZE: f32 = 0.02;     // 1マスを2cmとする

fn hash_position(p: vec3f) -> u32 {
    let grid_pos = vec3i(floor(p / CELL_SIZE));
    
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