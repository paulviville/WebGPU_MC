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
    @location(1) xyz : vec3f,
    @location(2) center : vec3f,
}

struct FragmentInput {
    @location(0) color: vec4f,
    @location(1) coord: vec3f,
    @location(2) center: vec3f,
}

struct FragmentOutput {
    @builtin(frag_depth) depth : f32,
    @location(0) color: vec4f,
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

const pointSize = 0.0125;
const lightPos : vec3f = vec3<f32>(1.0,1.0,1.0);

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var center = (uniforms.camera.view * vec4f(vertices[input.instance].position, 1.0)).xyz; 

    var position = input.position.xyz * pointSize;

    var color = vec4f(0.0, 0.0, 0.0, 1.0);
    if(vertices[input.instance].value > 0) {
        color.g = 1.0;
        position *= 0.0625;
    }
    else {
        color.r = 1.0;
    }
    position += center;
    
    return VertexOutput(
        uniforms.camera.projection * vec4f(position, 1.0),
        color,
        input.position.xyz,
        center,
    );
}

@fragment
fn fragment(input : FragmentInput) -> FragmentOutput {
    var N = vec3f(input.coord.x, input.coord.y, 1 - dot(input.coord.xy, input.coord.xy));
    if(N.z < 0.0) { discard; }
    
    N.z = sqrt(N.z);
    let P = input.center + N * pointSize;
    let L = normalize(lightPos - P);

    var output : FragmentOutput;
    output.color = vec4f(input.color.rgb * dot(N, L), 1.0);

    let wPos = uniforms.camera.projection * vec4f(P, 1.0);
    output.depth = (wPos.z / wPos.w + 1.0) / 2.0;
    return output;
}