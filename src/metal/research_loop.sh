#!/bin/bash
# Deep Research Loop for Apple Metal GPU
# Runs every 30 minutes to continue research iterations

cd /Users/longxia/Projects/GPUPeek/src/metal

LOG_FILE="/Users/longxia/Projects/GPUPeek/src/metal/research_loop.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Deep Research Loop Started ==="

while true; do
    log "--- Iteration Start ---"

    # Step 1: Build and run benchmark
    log "Building benchmark..."
    swift build --configuration release 2>&1 >> "$LOG_FILE"

    if [ $? -eq 0 ]; then
        log "Running benchmark..."
        swift run --configuration release 2>&1 >> "$LOG_FILE"
    else
        log "Build failed, skipping run"
    fi

    # Step 2: Check git status and commit if needed
    log "Checking for changes..."
    git status >> "$LOG_FILE" 2>&1

    # Step 3: Add and commit changes
    git add -A >> "$LOG_FILE" 2>&1
    if git diff --cached --quiet; then
        log "No changes to commit"
    else
        COMMIT_MSG="Deep research iteration $(date '+%Y%m%d_%H%M%S')"
        log "Committing: $COMMIT_MSG"
        git commit -m "$COMMIT_MSG" >> "$LOG_FILE" 2>&1

        log "Pushing..."
        git push >> "$LOG_FILE" 2>&1
        if [ $? -eq 0 ]; then
            log "Push successful"
        else
            log "Push failed (可能是没有 upstream 或网络问题)"
        fi
    fi

    log "--- Iteration Complete, sleeping 30 minutes ---"
    sleep 1800  # 30 minutes
done
