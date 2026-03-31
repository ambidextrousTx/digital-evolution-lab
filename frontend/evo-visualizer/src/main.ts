import * as THREE from 'three';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);

camera.position.set(0, 10, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(10, 10, 10);
scene.add(light);

const ambient = new THREE.AmbientLight(0x404040);
scene.add(ambient);

const size = 20;
const segments = 200;

const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
geometry.rotateX(-Math.PI / 2);

function fitness(x: number) {
  return x * Math.sin(10 * x) + x * Math.cos(2 * x);
}

const positions = geometry.attributes.position;

for (let i = 0; i < positions.count; i++) {
  const x = positions.getX(i);
  const z = positions.getZ(i);

  const y = fitness(x);

  positions.setY(i, y);
}

geometry.computeVertexNormals();

const material = new THREE.MeshStandardMaterial({
  color: 0x00ffcc,
  wireframe: false,
  side: THREE.DoubleSide,
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

const popSize = 100;

const positionsArray = new Float32Array(popSize * 3);

const geometryPoints = new THREE.BufferGeometry();
geometryPoints.setAttribute(
  'position',
  new THREE.BufferAttribute(positionsArray, 3)
);

const materialPoints = new THREE.PointsMaterial({
  color: 0xff5555,
  size: 0.2,
});

const points = new THREE.Points(geometryPoints, materialPoints);
scene.add(points);

function updatePopulation(genomes: number[]) {
  for (let i = 0; i < genomes.length; i++) {
    const x = genomes[i];
    const y = fitness(x);

    positionsArray[i * 3 + 0] = x;
    positionsArray[i * 3 + 1] = y + 0.2;
    positionsArray[i * 3 + 2] = 0;
  }

  geometryPoints.attributes.position.needsUpdate = true;
}

let mockPopulation = Array.from({ length: popSize }, () =>
  (Math.random() - 0.5) * 20
);

function evolveMockPopulation() {
  mockPopulation = mockPopulation.map(x => {
    // drift toward peak near ~6.4
    const target = 6.4;
    const drift = (target - x) * 0.02;
    const noise = (Math.random() - 0.5) * 0.3;

    return x + drift + noise;
  });

  updatePopulation(mockPopulation);
}

function animate() {
  requestAnimationFrame(animate);
  evolveMockPopulation();

  mesh.rotation.y += 0.002;

  renderer.render(scene, camera);
}

animate();
