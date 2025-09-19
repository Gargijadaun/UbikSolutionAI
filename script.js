let isTTSActive = true;
let scene, camera, renderer, model, mixer, animations = {};
let blendShapes = { eyeClosed: null, mouthOpen: null };
let isSpeaking = false;
let isRecording = false;
let recognition;

function initializeAvatar() {
    const canvas = document.getElementById('avatar');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);

    const loader = new THREE.GLTFLoader();
    loader.load(
        './models/AsianLadyCharacter_GLB.glb',
        (gltf) => {
            model = gltf.scene;
            scene.add(model);
            model.position.set(0, 0, 0);
            model.scale.set(1, 1, 1);

            // Initialize blend shapes for Asian_female_head
            model.traverse((child) => {
                if (child.isMesh && child.name === 'Asian_female_head' && child.morphTargetDictionary) {
                    blendShapes.eyeClosed = child.morphTargetDictionary['eyeClosed'] || 0;
                    blendShapes.mouthOpen = child.morphTargetDictionary['mouthOpen'] || 0;
                }
            });

            // Initialize animations
            mixer = new THREE.AnimationMixer(model);
            gltf.animations.forEach((clip) => {
                animations[clip.name] = mixer.clipAction(clip);
            });

            // Play Idle animation
            if (animations['Idle']) animations['Idle'].play();

            camera.position.z = 2;
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(0, 1, 0);
            controls.update();

            animate();
            document.getElementById('loading').style.display = 'none';
            if (document.querySelector('.tts-controls')) {
                document.querySelector('.tts-controls').style.display = 'block';
            }
        },
        (xhr) => {
            document.getElementById('loading').innerText = `Loading... ${Math.round(xhr.loaded / xhr.total * 100)}%`;
        },
        (error) => {
            console.error('Error loading model:', error);
            document.getElementById('loading').innerText = 'Error loading avatar';
        }
    );

    const light = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(light);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);
}

function animate() {
    requestAnimationFrame(animate);
    if (mixer) mixer.update(0.016);
    if (isSpeaking && blendShapes.mouthOpen !== null) {
        model.traverse((child) => {
            if (child.isMesh && child.name === 'Asian_female_head' && child.morphTargetInfluences) {
                child.morphTargetInfluences[blendShapes.mouthOpen] = Math.sin(Date.now() * 0.005) * 0.5 + 0.5;
                child.morphTargetInfluences[blendShapes.eyeClosed] = Math.random() < 0.05 ? 1 : 0; // Random blink
            }
        });
    }
    renderer.render(scene, camera);
}

function startExperience() {
    document.getElementById('welcome-screen').style.display = 'none';
    initializeAvatar();
}

function toggleRecording() {
    if (!isRecording) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.onresult = async (event) => {
            const transcript = event.results[0][0].transcript;
            addMessage(transcript, 'user');
            await sendMessage(transcript);
            toggleRecording();
        };
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            toggleRecording();
        };
        recognition.start();
        isRecording = true;
        document.querySelector('.voice-btn').classList.add('recording');
    } else {
        recognition.stop();
        isRecording = false;
        document.querySelector('.voice-btn').classList.remove('recording');
    }
}

async function sendMessage(input = null) {
    const baseURL = "http://localhost:8000";
    const questionInput = document.getElementById('question-input');
    const message = input || questionInput.value;
    if (!message.trim()) return;

    addMessage(message, 'user');
    questionInput.value = '';

    try {
        const response = await fetch(`${baseURL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'user1', question: message })
        });
        const data = await response.json();
        let responseText = data.answer || 'Error: No response';
        if (data.corrected) {
            addMessage(responseText, 'bot', true);
        } else {
            addMessage(responseText, 'bot');
        }
        if (isTTSActive) {
            await playTTS(responseText);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Error fetching response', 'bot');
    }
}

function addMessage(text, sender, corrected = false) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender} ${corrected ? 'corrected' : ''}`;
    messageDiv.innerHTML = `
        <div class="message-avatar">${sender === 'user' ? 'U' : 'B'}</div>
        <div class="message-content">${text}</div>
    `;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function playTTS(text) {
    try {
        const baseURL = window.location.pathname.includes('quiz.html') ? "http://localhost:10000" : "http://localhost:8000";
        isSpeaking = true;
        const talkingAnim = animations[Math.random() < 0.5 ? 'Talking1' : 'Talking2'];
        if (talkingAnim) talkingAnim.play();
        const response = await fetch(`${baseURL}/text-to-speech`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                language_code: 'en-GB',
                voice_name: 'en-GB-Standard-B'
            })
        });
        const data = await response.json();
        const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
        audio.play();
        audio.onended = () => {
            isSpeaking = false;
            if (talkingAnim) talkingAnim.stop();
            if (animations['Idle']) animations['Idle'].play();
        };
    } catch (error) {
        console.error('TTS error:', error);
        isSpeaking = false;
    }
}

function toggleTTS() {
    isTTSActive = !isTTSActive;
    document.querySelector('.control-btn').textContent = `TTS ${isTTSActive ? 'ON' : 'OFF'}`;
}