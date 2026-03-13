mod app;
mod engine;
mod renderer;
mod scene;
mod screenshot;
mod utils;
mod wgpu_ctx;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    // ログ出力用（wgpuのエラーなどを見るのに必須）
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    // 連続で描画を回し続けるためにPollを設定
    event_loop.set_control_flow(ControlFlow::Poll);

    // 固定解像度の定義を main.rs で行う
    let target_width = 1600;
    let target_height = 900;

    let mut app = app::App::new(target_width, target_height);
    event_loop.run_app(&mut app).unwrap();
}
