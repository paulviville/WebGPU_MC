console.log("WebGPU_MC")

// grid structure 
// grid order: along x, then y, then z;
//            6             7
//            +-------------+               +-----3-------+   
//          / |           / |             / |            /|   
//        /   |         /   |           6   10        7   11
//    4 +-----+-------+  5  |         +-----+2------+     |   
//      |   2 +-------+-----+ 3       |     +-----1-+-----+   
//      |   /         |   /           8   4         9   5
//      | /           | /             | /           | /       
//    0 +-------------+ 1             +------0------+         
//
// grid dimensions
const X = 64, Y = 64, Z = 64;
const nbVertices = (X+1) * (Y+1) * (Z+1);
const nbEdges = X*(Y+1)*(Z+1) + (X+1)*Y*(Z+1) + (X+1)*(Y+1)*Z;
const nbFaces = X*Y*(Z+1) + (X+1)*Y*Z + X*(Y+1)*Z;
const nbCubes = X*Y*Z;


// initializing webgpu
if(!navigator.gpu) throw Error("No GPU");

const adapter = await navigator.gpu.requestAdapter();
if(!adapter) throw Error("No adapter");

const device = await adapter.requestDevice();
if(!device) throw Error ("No device");

// console.log(device.limits)


// creating first compute 
const shaderCode = await fetch("./shaders/gridVertices.wgsl").then((response) => response.text())

const module = device.createShaderModule({
    label: 'first compute module',
    code: shaderCode,
});

const bindGroupLayout = device.createBindGroupLayout({
    label: 'first compute bindgroup layout',
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    }],
});

const pipeline = device.createComputePipeline({
    label: 'first compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
        module,
        entryPoint: "main",
        constants: {
            X: X,
            Y: Y,
            // Z: Z,
        }
    },
});

const vertexStorageBuffer = device.createBuffer({
    label: "vertex position storage buffer",
    size: nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const vertexStagingBuffer = device.createBuffer({
    label: "vertex position staging buffer",
    size: nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const bindGroup = device.createBindGroup({
    label: 'first compute bind group',
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: {
            buffer: vertexStorageBuffer,
        },
    }],
});



const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();

passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(Math.ceil(nbVertices / 64));
passEncoder.end();

console.log(Math.ceil(nbVertices / 64))

commandEncoder.copyBufferToBuffer(
    vertexStorageBuffer,
    0, //offset
    vertexStagingBuffer,
    0, // offset
    nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT   
);

const commands = commandEncoder.finish();
device.queue.submit([commands]);

await vertexStagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, //offset
    nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT
);
const copyArrayBuffer = vertexStagingBuffer.getMappedRange(0, nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT);
const data = copyArrayBuffer.slice();
console.log(copyArrayBuffer.slice())
vertexStagingBuffer.unmap();
// console.log(new Float32Array(data))
const testArray = new Float32Array(data)
// console.log(nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT)
// console.log(testArray.length)

// function vertId (i) {
//     let id = i;
//     let z = id / ((X+1))
// }

// for(let i = 0; i < testArray.length; i += 4) {
//     console.log(testArray[i], testArray[i+1], testArray[i+2]);
// }