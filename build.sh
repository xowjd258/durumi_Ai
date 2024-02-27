#!/bin/bash

# .env 파일에서 환경 변수를 로드
export $(cat .env | xargs)

# Docker 이미지를 빌드
docker build --no-cache --build-arg OPENAI_API_KEY=$OPENAI_API_KEY  -t durumi .
