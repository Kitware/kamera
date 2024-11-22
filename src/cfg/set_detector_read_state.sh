# Source top-level env file

SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)
KAM_REPO_DIR=$(/home/user/.config/kamera/repo_dir.bash)
source "${KAM_REPO_DIR}/tmux/${SYSTEM_NAME}/env.sh"

# Pull from .env, and set globally in Redis
echo "Setting global redis params."
echo "Setting read_from_nas to ${READ_FROM_NAS}."
redis-cli -h ${REDIS_HOST} set "/detector/read_from_nas" "${READ_FROM_NAS}"
echo "Setting compress_imagery to ${COMPRESS_IMAGERY}."
redis-cli -h ${REDIS_HOST} set "/imageview/compress_imagery" "${COMPRESS_IMAGERY}"
echo "Setting jpeg_quality to ${JPEG_QUALITY}."
redis-cli -h ${REDIS_HOST} set "/sys/arch/jpeg/quality" "${JPEG_QUALITY}"
echo "Setting spoof_events to ${SPOOF_EVENTS}."
redis-cli -h ${REDIS_HOST} set "/debug/spoof_events" "${SPOOF_EVENTS}"
# These keys are needed to allow spoof events to work
redis-cli -h ${REDIS_HOST} set "/debug/enable" 1
redis-cli -h ${REDIS_HOST} set "/debug/trigger_pps" 0
