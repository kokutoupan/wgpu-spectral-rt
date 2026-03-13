use crate::engine::Engine;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

pub struct App {
    engine: Option<Engine>,
    target_width: u32,
    target_height: u32,
}

impl App {
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            engine: None,
            target_width,
            target_height,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.engine.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("wgpu-spectral-rt")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.target_width as f64,
                    self.target_height as f64,
                ));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

            // wgpuの初期化は非同期なので、pollsterでブロックして待つ
            self.engine = Some(pollster::block_on(Engine::new(
                window,
                self.target_width,
                self.target_height,
            )));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };

        engine.input(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                engine.resize(physical_size);
                engine.window().request_redraw();
            }
            // 描画のタイミング
            WindowEvent::RedrawRequested => {
                engine.update();
                match engine.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        engine.resize(engine.window().inner_size());
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
                // 次のフレームを描画するように要求（これでループが回り続ける）
                engine.window().request_redraw();
            }
            _ => (),
        }
    }
}
