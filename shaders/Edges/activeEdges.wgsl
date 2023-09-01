struct Vertex {
    position: vec3f,
    value: f32,
}

@group(0) @binding(0)
var <storage, read> vertices : array<Vertex>;
@group(0) @binding(1)
var <storage, read_write> edgesId : array<u32>;
@group(0) @binding(2)
var<storage, read_write> chunkEdges : array<u32>;
@group(0) @binding(3)
var<storage, read_write> edgeChunkOffset : array<u32>;
@group(0) @binding(4)
var<storage, read_write> edgeMid : array<vec3f>;

const INVALID = 0u - 1u;
const Min : vec3f = vec3f(-1.0,-1.0,-1.0);
const Max : vec3f = vec3f(1.0,1.0,1.0);
override STEP : f32 = 2.0 / 16.0;
override X : u32 = 16u;
override Y : u32 = 16u;
override Z : u32 = 16u;
override NbVertices = (X+1) * (Y+1) * (Z+1);
override CHUNKSIZE = 64u;
//const step : vec3f = vec3f

fn computeVertexId(globalId: u32) -> vec3u {
    var id = globalId;
    let z = id / ((X+1)*(Y+1));
    id %= ((X+1)*(Y+1));
    let y = id / (X+1);
    let x = id % (X+1);
    return vec3u(x, y, z);
}

fn computeVertexValue(pos: vec3f) -> f32 {
    // www-sop.inria.fr/galaad/surface/
    // chubs
    return pow(pos.x, 4)+pow(pos.y, 4)+pow(pos.z, 4)-pow(pos.x, 2)-pow(pos.y, 2)-pow(pos.z, 2)+0.5; 
}

fn vertexId(coord: vec3u) -> u32{
    return coord.x + (X+1) * coord.y + (X+1)*(Y+1) * coord.z;
}



override nbX = (X)*(Y+1)*(Z+1);
override nbY = (X+1)*(Y)*(Z+1);
override nbZ = (X+1)*(Y+1)*(Z);

fn edgeCell(e: u32) -> vec3u {
    let ABC = vec3u(
        u32(e >= nbX) * nbX + u32(e >= (nbX + nbY)) * nbY,
        (X + u32(e > nbX)) * (Y + u32((e < nbX) || (e > nbX + nbY))),
        X + u32(e > nbX)
    );

    var xyz : vec3u;
    var e_ = e - ABC.x;
    xyz.z = e_ / ABC.y;
    e_ = e_ % ABC.y;
    xyz.y = e_ / ABC.z;
    xyz.x = e_ % ABC.z;

    return xyz;
}

@compute @workgroup_size(64)
fn markActiveEdges(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)) {
        return;
    }

    let cell = edgeCell(global_id.x);

    let axis = vec3u(
        u32(global_id.x < nbX),
        u32((global_id.x < nbY + nbX) && global_id.x >= nbX),
        u32(global_id.x >= (nbX + nbY))
    );

    let v0 = vertices[vertexId(cell)].value;
    let v1 = vertices[vertexId(cell+axis)].value;

    if((v0 * v1) <= 0.0) {
        edgesId[global_id.x] = global_id.x; 
    } else {
        edgesId[global_id.x] = INVALID;
    }
}

@compute @workgroup_size(64)
fn countEdgesPerChunk(@builtin(global_invocation_id) global_id : vec3<u32>) {
    _ = X;
    _ = Y;
    _ = Z;

    /// first edge of chunk
    let chunkOffset = global_id.x * CHUNKSIZE;
    var localOffset = 0u;
    for(var i = 0u; i < CHUNKSIZE; i++) {
        if(edgesId[chunkOffset + i] != INVALID) {
            edgeChunkOffset[chunkOffset + i] = localOffset;
            localOffset++;
        } else {
            edgeChunkOffset[chunkOffset + i] = INVALID;
        }
    }

    chunkEdges[global_id.x] = localOffset;
}

@compute @workgroup_size(1)
fn reduceEdgeCount() {
    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;

    var off0 = chunkEdges[0];
    var off1 = 0u;
    chunkEdges[0] = 0;
    for(var i = 0u; i < arrayLength(&chunkEdges) - 1; i++) {
        off1 = chunkEdges[i+1];
        chunkEdges[i+1] = chunkEdges[i] + off0;
        off0 = off1;
    }
}

@compute @workgroup_size(64)
fn completeEdgeOffsets(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)
        || edgesId[global_id.x] == INVALID) {
        return;
    }

    let chunkId = global_id.x / CHUNKSIZE;
    let chunkOffset = chunkEdges[chunkId];

    edgeChunkOffset[global_id.x] += chunkOffset ;
}

@compute @workgroup_size(64)
fn computEdgeMid(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)
        || edgesId[global_id.x] == INVALID) {
        return;
    }

    let cell = edgeCell(global_id.x);

    let axis = vec3u(
        u32(global_id.x < nbX),
        u32((global_id.x < nbY + nbX) && global_id.x >= nbX),
        u32(global_id.x >= (nbX + nbY))
    );

    let v0 = vertices[vertexId(cell)].position;
    let v1 = vertices[vertexId(cell+axis)].position;

    let offset = edgeChunkOffset[global_id.x];
    edgeMid[offset] = (v0 + v1) / 2.0;
}
