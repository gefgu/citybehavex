# TEI server setup commands (run these yourself, they need sudo password)

## 1. Remove the previous failed container and start fresh with CDI device syntax

```
sudo docker rm -f tei-schedule-aligner 2>/dev/null; sudo docker run -d --device nvidia.com/gpu=all --name tei-schedule-aligner -p 8082:80 -v /home/gustavo/citybehavex/models/modernbert-schedule-aligner/:/data/model ghcr.io/huggingface/text-embeddings-inference:120-1.9 --model-id /data/model
```

## 2. If that still errors, fall back to the legacy nvidia runtime flag

```
sudo docker rm -f tei-schedule-aligner 2>/dev/null; sudo docker run -d --runtime=nvidia --name tei-schedule-aligner -p 8082:80 -v /home/gustavo/citybehavex/models/modernbert-schedule-aligner/:/data/model ghcr.io/huggingface/text-embeddings-inference:120-1.9 --model-id /data/model
```

## 3. Check status / logs after running either command above

```
sudo docker ps -a --filter name=tei-schedule-aligner
sudo docker logs tei-schedule-aligner
```
