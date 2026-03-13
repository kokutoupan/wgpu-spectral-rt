mod app;
mod engine;
mod renderer;
mod scene;
mod screenshot;
mod wgpu_ctx;
mod utils;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    // ログ出力用（wgpuのエラーなどを見るのに必須）
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    // 連続で描画を回し続けるためにPollを設定
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::default();
    event_loop.run_app(&mut app).unwrap();
}
