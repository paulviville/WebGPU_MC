struct Vertex {
    position: vec3f,
    value: f32,
}

struct VertexInput {
    @builtin(instance_index) instance : u32,
    @location(0) position : vec3f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f,
}

struct Camera {
    @size(16) position: vec3f,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

struct Uniforms {
    camera: Camera,
    //model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var <storage, read> vertices : array<Vertex>;

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var position = input.position.xyz * 0.0125;
    position += vertices[input.instance].position;
    var color = vec4f(0.0, 0.0, 0.0, 1.0);
    if(vertices[input.instance].value > 0) {
        color.g = 1.0;
    }
    else {
        color.r = 1.0;
    }
    return VertexOutput(
        uniforms.camera.projection * uniforms.camera.view * vec4f(position, 1.0),
        color,
    );
}

@fragment
fn fragment(@location(0) color: vec4f) -> @location(0) vec4f {
    if(color.g > 0.0) {
        discard;
    }
    return color;
}