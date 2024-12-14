<template>
  <div id="app-section">
    <div class="section" id="section-0">
      <div>Scanner 3D</div>
      <button id="pair-button" @click="updateIpStream">
        <img src="https://res.cloudinary.com/dozfohnhs/image/upload/v1734110381/Online_kryeqw.png" alt="">
        <p>Pair device</p>
      </button>
      <button id="scan-button" @click="startScan">
        <img src="https://res.cloudinary.com/dozfohnhs/image/upload/v1734110381/Shutdown_uawsjq.png" alt="">
        <p>Start Scan</p>
      </button>
    </div>
    <div class="section" id="section-1">
      <img id="stream" :src="urlStream" ref="frame" alt="" crossorigin="anonymous">
      <div id="tag-live">
        <img src="https://res.cloudinary.com/dozfohnhs/image/upload/v1734110387/Youtube_Live_kyptvu.png">
        <p id="current-ip">{{ currentIp }}</p>
      </div>
    </div>
    <div class="section" id="section-2">
      <div id="state-scan" v-if="!glbUrl">
        <p id="state"> {{ state }} </p>
      </div>
      <div id="model"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const currentIp = ref('Not Ip');
const urlStream = ref('https://res.cloudinary.com/dozfohnhs/image/upload/v1734113052/1iewg3rme8d61.png_tjbtkz.webp');
const backendUrl = "http://localhost:8000";
const state = ref("Not model available");
const glbUrl = ref("");
const frame = ref();
const isScanning = ref(false);
let intervalId = null;

const updateIpStream = async () => {
  try {
    const res = await fetch(`${backendUrl}/scan/stream`);
    if (res.ok) {
      const data = await res.json();
      currentIp.value = data.ip;
      urlStream.value = `http://${data.ip}:81/stream`;
      console.log(`URL stream: ${urlStream.value}`);
    } else {
      console.error("Error fetching IP stream:", res.statusText);
    }
  } catch (error) {
    console.error("Error updating IP stream:", error);
  }
};

const sendFrame = async () => {
  const frameElement = frame.value;
  if (!frameElement) {
    console.error("Frame element not found.");
    return;
  }

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 635;
  canvas.height = 471;

  ctx.drawImage(frameElement, 0, 0);

  canvas.toBlob(async (blob) => {
    if (!blob) {
      console.error("Error converting image to Blob.");
      return;
    }
    const formData = new FormData();
    formData.append("image", blob, "imagen.jpg");
    try {
      const response = await fetch(`${backendUrl}/scan/load`, {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        console.log("Image sent successfully:", data);
      } else {
        console.error("Server error:", response.statusText);
      }
    } catch (error) {
      console.error("Error sending image:", error);
    }
  }, "image/jpeg");
};

const startScan = async () => {
  state.value = "Generating 3D model...";
  isScanning.value = true;

  // Iniciar el envío periódico de frames
  intervalId = setInterval(() => {
    if (isScanning.value) {
      sendFrame();
    }
  }, 100); // Captura y envío cada 500ms (ajustable según sea necesario)

  try {
    const res = await fetch(`${backendUrl}/scan/start`);
    if (!res.ok) throw new Error(`Start scan failed: ${res.statusText}`);

    // Termina el escaneo cuando se reciba la señal del backend
    isScanning.value = false;
    clearInterval(intervalId);

    const processRes = await fetch(`${backendUrl}/scan/process`);
    const glbBlob = await processRes.blob();
    glbUrl.value = URL.createObjectURL(glbBlob);
    console.log("3D model ready:", glbUrl.value);
    state.value = "3D model generated!";
  } catch (error) {
    console.error("Error generating 3D model:", error);
    state.value = "Error generating 3D model.";
    isScanning.value = false;
    clearInterval(intervalId);
  }
};

const renderModel = () => {
  const container = document.getElementById('model');
  if (!container) return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    75,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.innerHTML = ""; // Clear before adding canvas
  container.appendChild(renderer.domElement);

  const light = new THREE.AmbientLight(0xffffff, 1);
  scene.add(light);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
  directionalLight.position.set(5, 5, 5);  // Posición de la luz
  scene.add(directionalLight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  camera.position.set(2, 2, 5);

  const loader = new GLTFLoader();
  loader.load(
    glbUrl.value,
    (gltf) => {
      const model = gltf.scene;
      scene.add(model);

      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      model.position.sub(center);

      const size = box.getSize(new THREE.Vector3());
      model.scale.setScalar(1 / size.length());
    },
    undefined,
    (error) => console.error("Error loading GLB model:", error)
  );

  const animate = () => {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };

  animate();
};

watch(glbUrl, (newUrl) => {
  if (newUrl) {
    renderModel();
  }
});
</script>

<style>
* {
  padding: 0;
  margin: 0;
}

body {
  width: 100%;
  height: 100dvh;
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  box-sizing: border-box;
}

#app-section {
  width: auto;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 10px
}

.section {
  width: auto;
  max-width: 430px;
  height: auto;
  background: white;
  font-family: "Fira Code", monospace;
}

/* Section head */

#section-0 {
  height: 43px;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

#section-0 div {
  height: 100%;
  background: #4653E1;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  padding: 0 20px 0 20px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  color: #fff;
  font-weight: bold;
}

#section-0 button {
  height: 100%;
  display: flex;
  flex-direction: row;
  gap: 10px;
  justify-content: center;
  align-items: center;
  box-sizing: border-box;
  padding: 0 20px 0 20px;
  border: 1px solid #000;
  background: #fff;

  font-size: 13px;
  font-weight: 500;
  border-radius: 8px;

  cursor: pointer;
}

#section-0 button:hover {
  background: #F5F5F5;
}

/* Section video */

#section-1 {
  position: relative;
  width: auto;
  height: 273px;
  background: #fff;
  border: 1px solid #000;
  border-radius: 20px;
  padding: 10px;
  box-sizing: border-box;
  overflow: hidden;
}

#section-1 img {
  width: 100%;
  height: 100%;
  border-radius: 8px;
}

#section-1 #tag-live {
  position: absolute;
  top: 20px;
  left: 20px;
  width: 94px;
  height: 23px;
  background: #fff;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 0 8px 0 8px;
  border-radius: 4px;
  font-size: 11px;
}

#section-1 #tag-live img {
  height: 15px;
  width: 15px;
}

/* Section model */

#section-2 {
  width: auto;
  height: 370px;
  background: #fff;
  border: 1px solid #A7A7A7;
  border-radius: 20px;
  padding: 10px;
  box-sizing: border-box;
  overflow: hidden;
}

#section-2 #state-scan {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: #F5F5F5;
  border-radius: 10px;
  font-size: 11px;
}

#model {
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
  border-radius: 10px;
}
</style>
