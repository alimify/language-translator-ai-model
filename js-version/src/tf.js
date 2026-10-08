import * as tf from "@tensorflow/tfjs";

let backendName = "cpu";

async function configureTensorFlow() {
    try {
        await import("@tensorflow/tfjs-node");
        await tf.setBackend("tensorflow");
        backendName = "tensorflow";
    } catch {
        await tf.setBackend("cpu");
        backendName = "cpu";
    }
    await tf.ready();
    console.log(`TensorFlow.js backend: ${backendName}`);
}

await configureTensorFlow();

export { tf, backendName };
