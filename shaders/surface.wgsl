struct Vertex {
    position: vec3f,
    value: f32,
}

struct Camera {
    @size(16) position: vec3f,
	view : mat4x4<f32>,
	projection: mat4x4<f32>,
}

struct VertexInput {
    @builtin(vertex_index) vid : u32,
    @location(0) position : vec4f,
    // @location(1) normal : vec3f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) normal : vec3f,
}

struct FragmentInput {
    @location(0) position : vec3f,
}

struct FragmentOutput {
    @location(0) color : vec4f,
}

struct Uniforms {
	camera: Camera,
}


@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var <storage, read> vertices : array<Vertex>;

@vertex
fn vertex(input : VertexInput) -> VertexOutput {
    let position = vec4f(input.position.xyz, 1);
	// let position = vec4f(vertices[input.vid].position, 1);
    // let normal = ( vec4f(input.normal, 0)).xyz;
	let normal = (uniforms.camera.view * position).xyz;
    return VertexOutput(
        uniforms.camera.projection * uniforms.camera.view * position,
        normal.xyz,
    );
}

const lightPos : vec3f = vec3<f32>(1.0,1.0,-13.0);
@fragment
fn fragment(input : FragmentInput) -> FragmentOutput {
	var N : vec3f;
	N = normalize(cross(dpdx(input.position), dpdy(input.position)));
    // N = input.position;
	let P = input.position;
    let L = normalize(lightPos - P);
    // let L = normalize((uniforms.camera.view * vec4f(lightPos, 1.0)).xyz - P);
    return FragmentOutput(
        // vec4f(N, 1.0),
        vec4f(vec3f(max(dot(N, L), 0.2)), 1.0),
        // vec4f(vec3f(max(dot(N, L), 0.2*dot(N, -L))), 1.0),
    );
}
