use crate::scene::builder::{MaterialUniform, SceneBuilder, SceneResources};
use crate::scene::geometry;
use glam::{Mat4, Vec3};

#[allow(dead_code)]
pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    let mat_light = builder.add_material(MaterialUniform {
        color: [6500.0, 5.0, 0.0, 1.0],
        extra: [3., 0., 0., 0.],
    });
    let mat_red = builder.add_material(MaterialUniform {
        color: [0.65, 0.05, 0.05, 1.],
        extra: [0., 0., 0., 0.],
    });
    let mat_green = builder.add_material(MaterialUniform {
        color: [0.12, 0.45, 0.15, 1.],
        extra: [0., 0., 0., 0.],
    });
    let mat_white = builder.add_material(MaterialUniform {
        color: [0.73, 0.73, 0.73, 1.],
        extra: [0., 0., 0., 0.],
    });
    let mat_glass = builder.add_material(MaterialUniform {
        color: [1., 1., 1., 1.],
        extra: [2., 0., 1.5, 0.02],
    });
    let mat_metal = builder.add_material(MaterialUniform {
        color: [0.8, 0.8, 0.8, 1.],
        extra: [1., 0.0, 0., 0.],
    });

    let (plane_v, plane_i) = geometry::create_plane();
    let (cube_v, cube_i) = geometry::create_cube();
    let (sphere_v, sphere_i) = geometry::create_sphere(4);

    let mesh_plane = builder.add_mesh(&plane_v, &plane_i);
    let mesh_cube = builder.add_mesh(&cube_v, &cube_i);
    let mesh_sphere = builder.add_mesh(&sphere_v, &sphere_i);

    builder.add_instance(
        mesh_plane,
        mat_white,
        Mat4::from_translation(Vec3::new(0., -1., 0.)) * Mat4::from_scale(Vec3::splat(2.)),
    );
    builder.add_instance(
        mesh_plane,
        mat_white,
        Mat4::from_translation(Vec3::new(0., 1., 0.))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.)),
    );
    builder.add_instance(
        mesh_plane,
        mat_white,
        Mat4::from_translation(Vec3::new(0., 0., -1.))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.)),
    );
    builder.add_instance(
        mesh_plane,
        mat_red,
        Mat4::from_translation(Vec3::new(-1., 0., 0.))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.)),
    );
    builder.add_instance(
        mesh_plane,
        mat_green,
        Mat4::from_translation(Vec3::new(1., 0., 0.))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.)),
    );
    // builder.add_instance(
    //     mesh_plane,
    //     mat_light,
    //     Mat4::from_translation(Vec3::new(0., 0.99, 0.))
    //         * Mat4::from_rotation_x(std::f32::consts::PI)
    //         * Mat4::from_scale(Vec3::splat(0.5)),
    // );

    builder.add_light_instance(
        mesh_plane,
        mat_light,
        Mat4::from_translation(Vec3::new(0., 0.99, 0.))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
        [0.0, 6500.0, 5.0, 0.0], // params: type=0, temp=6500, intensity=15
    );

    builder.add_instance(
        mesh_cube,
        mat_metal,
        Mat4::from_translation(Vec3::new(-0.35, -0.4 + 0.002, -0.3))
            * Mat4::from_rotation_y(0.4)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
    );
    builder.add_instance(
        mesh_sphere,
        mat_glass,
        Mat4::from_translation(Vec3::new(0.4, -0.65, 0.3)) * Mat4::from_scale(Vec3::splat(0.75)),
    );

    builder.build(device, queue)
}
