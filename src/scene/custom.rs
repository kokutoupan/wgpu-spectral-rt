use crate::scene::builder::{MaterialUniform, SceneBuilder, SceneResources};
use crate::scene::geometry;
use glam::{Mat4, Vec3};

#[allow(dead_code)]
pub fn create_custom_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // Scene Materials
    let mat_light = builder.add_material(MaterialUniform {
        color: [6500.0, 5.0, 0.0, 1.0], // Super high intensity vertical slit
        extra: [3., 0., 0., 0.],
    });

    // Pitch black walls to isolate the light streak completely
    let mat_wall = builder.add_material(MaterialUniform {
        color: [0.01, 0.01, 0.01, 1.],
        extra: [0., 0., 0., 0.],
    });

    let mat_canvas_board = builder.add_material(MaterialUniform {
        color: [0.95, 0.95, 0.95, 1.],
        extra: [0., 0., 0., 0.],
    });

    let mat_prism_glass = builder.add_material(MaterialUniform {
        color: [1., 1., 1., 1.],
        extra: [2., 0.0, 1.6, 0.12], // Max dispersion for maximum rainbow spread
    });

    // bubble
    let mat_bubble = builder.add_material(MaterialUniform {
        color: [1.0, 1.0, 1.0, 1.],
        extra: [4., 500.0, 1.33, 1.0], // perfectly smooth
    });

    // Geometries
    let (plane_v, plane_i) = geometry::create_plane();
    let (cube_v, cube_i) = geometry::create_cube();
    let (sphere_v, sphere_i) = geometry::create_sphere(4);

    let mesh_plane = builder.add_mesh(&plane_v, &plane_i);
    let mesh_cube = builder.add_mesh(&cube_v, &cube_i);
    let mesh_sphere = builder.add_mesh(&sphere_v, &sphere_i);

    let room_size = 4.0;

    // ----- ROOM -----
    // Floor
    builder.add_instance(
        mesh_plane,
        mat_wall,
        Mat4::from_translation(Vec3::new(0., -1., 0.))
            * Mat4::from_scale(Vec3::splat(room_size * 2.0)),
    );
    // Ceiling
    builder.add_instance(
        mesh_plane,
        mat_wall,
        Mat4::from_translation(Vec3::new(0., room_size - 1.0, 0.))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(room_size * 2.0)),
    );
    // Back Wall (The rainbow might hit this too)
    builder.add_instance(
        mesh_plane,
        mat_wall,
        Mat4::from_translation(Vec3::new(0., room_size * 0.5 - 1.0, -room_size))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(room_size * 2.0)),
    );

    // ----- THE EXPERIMENT SETUP -----

    // 1. The Canvas Board (A long table where the light streaks across)
    builder.add_instance(
        mesh_cube,
        mat_canvas_board,
        Mat4::from_translation(Vec3::new(0.5, -0.55, 0.0))
            * Mat4::from_scale(Vec3::new(5.0, 0.1, 3.0)),
    );

    // 2. The Vertical Slit Light Box
    // Placed at the left (-X), pointing right (+X)
    // Scale: Small Z (narrow width), large X (which becomes Y height after rotation)
    builder.add_light_instance(
        mesh_plane,
        mat_light,
        Mat4::from_translation(Vec3::new(-2.0, -0.2, 0.0))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2 + 0.02) // Point +X, slightly angled down
            * Mat4::from_scale(Vec3::new(0.5, 1.0, 0.02)), // Vertical slit (0.8 tall, 0.03 wide)
        [0.0, 6500.0, 5.0, 0.0],
    );

    // (Removed the dim ceiling light completely for a dark room)

    // 3. The Prism
    // Rotated cube around Y axis so the vertical edge faces the light,
    // causing a horizontal dispersion of the spectrum across the table.
    builder.add_instance(
        mesh_cube,
        mat_prism_glass,
        Mat4::from_translation(Vec3::new(-1.1, -0.2, 0.0))
            * Mat4::from_rotation_y(0.7) // Tilt to refract
            * Mat4::from_scale(Vec3::new(0.5, 0.8, 0.5)), // Tall prism
    );

    // 4. A Chrome sphere in the distance to catch nice reflections
    builder.add_instance(
        mesh_sphere,
        mat_bubble,
        Mat4::from_translation(Vec3::new(1.0, -0.1, -0.5)) * Mat4::from_scale(Vec3::splat(0.6)),
    );

    builder.build(device, queue)
}
