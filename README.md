# WebWhisper

Transcrição de áudio 100% local no navegador, usando [Transformers.js v3](https://huggingface.co/docs/transformers.js) com modelo Whisper rodando em **WebGPU** (com fallback para WebAssembly).

Sem API, sem servidor, sem custo. O modelo é baixado uma vez (~290 MB) e fica em cache no navegador (IndexedDB).

## Como funciona

1. Clique no botão de microfone → permissão de áudio é solicitada.
2. O modelo Whisper é carregado na primeira vez (mostra progresso).
3. A gravação começa. Clique de novo para parar.
4. O áudio é decodificado para PCM 16 kHz mono e enviado para o pipeline `automatic-speech-recognition` rodando localmente.
5. O resultado aparece como um card no topo da tela; novas gravações empilham acima.
6. Botão de copiar em cada card.

Atalho: `Espaço` inicia/para a gravação.

## Rodando localmente

```bash
npm install
npm run dev
```

A página abre em `http://localhost:5173` (ou similar). Em `localhost`, o navegador permite acesso ao microfone sem HTTPS.

## Deploy no Firebase Hosting

Pré-requisitos: ter o `firebase-tools` instalado e estar logado.

```bash
npm install -g firebase-tools     # se ainda não tiver
firebase login
```

Editar `.firebaserc` e trocar `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID` pelo seu Project ID do Firebase.

Build + deploy:

```bash
npm run deploy
```

Isso roda `vite build` e publica o conteúdo de `dist/`.

### Headers importantes (já configurados em `firebase.json`)

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

Esses headers ativam **cross-origin isolation**, necessária para WebGPU/threads. Sem eles, o modelo pode falhar ou rodar muito devagar.

## Trocar o modelo

Em `src/main.js`, mudar `MODEL_ID`:

- `Xenova/whisper-tiny` — ~150 MB, mais rápido, qualidade menor.
- `Xenova/whisper-base` — ~290 MB, equilíbrio (padrão).
- `Xenova/whisper-small` — ~970 MB, melhor qualidade, mais lento.

## Stack

- Vite (build)
- @huggingface/transformers v3 (Whisper local)
- Lucide (ícones)
- MediaRecorder API + Web Audio API (gravação e resampling)
- Firebase Hosting (estático)
