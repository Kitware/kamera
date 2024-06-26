# Source top-level env file
source ~/.config/kamera/repo_dir.bash
source ${KAM_REPO_DIR}/tmux/cas/env.sh

# Pull from .env, and set globally in Redis
echo "Setting global redis params."
redis-cli -h ${REDIS_HOST} set "/detector/read_from_nas" "${READ_FROM_NAS}"
redis-cli -h ${REDIS_HOST} set "/imageview/compress_imagery" "${COMPRESS_IMAGERY}"
redis-cli -h ${REDIS_HOST} set "/sys/arch/jpeg/quality" "${JPEG_QUALITY}"
redis-cli -h ${REDIS_HOST} set "/debug/spoof_events" "${SPOOF_EVENTS}"
