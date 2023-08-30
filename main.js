import { Loader } from "./utils/loader.js";
import { Vertex } from "./utils/mesh.js";
import { Model } from "./utils/model.js";
import { OrbitCamera } from "./utils/orbit-camera.js";

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
const X = 16, Y = 16, Z = 16;
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
//     console.log(`(${testArray[i]}, ${testArray[i+1]}, ${testArray[i+2]}) -> ${testArray[i+3].toPrecision(2)}`)
// }






















const canvas = document.getElementById('webGpuCanvas');
const context = canvas.getContext('webgpu');
context.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
});


const camera = new OrbitCamera(canvas);

/// create spheres
const assetLoader = new Loader;
const model = new Model(await assetLoader.loadModel('utils/billboard.json'));
const vertexBuffer = model.createVertexBuffer(device);
const indexBuffer = model.createIndexBuffer(device);


// create rendering resources
const uniformBuffer = device.createBuffer({
    label: 'sphere uniform buffer',
    size: 144,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const depthTexture = device.createTexture({
    label: 'depth texture',
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});




const gridShaderCode = await fetch("./shaders/Vertices/vertexBillboard.wgsl").then((response) => response.text())
const gridShaderModule = device.createShaderModule({
    label: 'grid vertex shader',
    code: gridShaderCode,
});

const gridBindGroupLayout = device.createBindGroupLayout({
    label: 'grid bind group layout',
    entries: [
        { /// uniform buffer
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUBufferUsage.FRAGMENT,
            buffer: {},
        },
        { /// vertex storage buffer
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUBufferUsage.FRAGMENT,
            buffer: {
                type: 'read-only-storage',
            }
        }
    ]
});

const gridPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [
        gridBindGroupLayout,
    ],
});

const gridPipeline = device.createRenderPipeline({
    label: 'grid pipeline',
    layout: gridPipelineLayout,
    vertex: {
        module: gridShaderModule,
        entryPoint: 'vertex',
        buffers: [Vertex.vertexLayout()],
    },
    fragment: {
        module: gridShaderModule,
        entryPoint: 'fragment',
        targets: [{
            format: navigator.gpu.getPreferredCanvasFormat(),
        }]
    },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
    },
    primitive: {
        cullMode: 'back',
        // topology: "point-list"
    },
});

const gridBindGroup = device.createBindGroup({
    label: 'grid bind group',
    layout: gridBindGroupLayout,
    entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: vertexStorageBuffer}},
    ],
});

const colorAttachment = {
    view: null,
    clearValue: {r: 0, g: 0, b: 0, a: 1},
    loadOp: 'clear',
    loadValue: {r: 0, g: 0, b: 0, a: 1},
    storeOp: 'store',
};

const depthStencilAttachment = {
    view: depthTexture.createView(),
    depthClearValue: 1.0,
    depthLoadOp: 'clear',
    depthStoreOp: 'discard',
};


let skip = false;
/// rendering
function render() {

    if(!skip){
    //move to render function
    const uniformArray = new Float32Array([
        ...camera.position, 0.0,
        ...camera.view,
        ...camera.projection,
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    camera.update();
    colorAttachment.view = context.getCurrentTexture().createView();
    const renderCommandEncoder = device.createCommandEncoder();
    const renderPass = renderCommandEncoder.beginRenderPass({
        colorAttachments: [colorAttachment],
        depthStencilAttachment: depthStencilAttachment,
    });
    renderPass.setPipeline(gridPipeline);
    renderPass.setBindGroup(0, gridBindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, model.indexType);
    renderPass.drawIndexed(model.numIndices, nbVertices);
    renderPass.end();

    device.queue.submit([renderCommandEncoder.finish()]);
    }
    skip = !skip;
    requestAnimationFrame(render)
}
render();