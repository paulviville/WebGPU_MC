struct Vertex {
    position: vec3f,
    value: f32,
}

@group(0) @binding(0)
var <storage, read> vertices : array<Vertex>;
@group(0) @binding(1)
var<storage, read_write> edgeChunkOffset : array<u32>;
@group(0) @binding(2)
var<storage, read_write> rawTri : array<u32>;
@group(0) @binding(3)
var<storage, read_write> indexCount : array<u32>;
@group(0) @binding(4)
var<storage, read_write> rawTriOffset : array<u32>;
@group(0) @binding(5)
var<storage, read_write> indexBuffer : array<u32>;

const INVALID = 0u - 1u;
const inv = INVALID;
const Min : vec3f = vec3f(-1.0,-1.0,-1.0);
const Max : vec3f = vec3f(1.0,1.0,1.0);
override STEP : f32 = 2.0 / 16.0;
override X : u32 = 16u;
override Y : u32 = 16u;
override Z : u32 = 16u;
override NbVertices = (X+1) * (Y+1) * (Z+1);
override CHUNKSIZE = 64u;
//const step : vec3f = vec3f



@compute @workgroup_size(64)
fn copyToIndexBuffer(@builtin(global_invocation_id) global_id : vec3<u32>) {
	if(global_id.x >= arrayLength(&rawTriOffset)
        || rawTriOffset[global_id.x] == INVALID) {
        return;
    }

	indexBuffer[rawTriOffset[global_id.x]] = rawTri[global_id.x];

    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;
}