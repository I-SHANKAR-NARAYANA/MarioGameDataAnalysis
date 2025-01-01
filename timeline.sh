#!/bin/bash
# Rewrite all commit dates (except the very first one) so they fall
# sequentially between Jan 1, 2025 and Feb 28, 2025.

set -e

# Move to repo root
cd "$(git rev-parse --show-toplevel)"

# Count commits
total=$(git rev-list --count HEAD)
first_commit=$(git rev-list --max-parents=0 HEAD)

# We need new dates for all commits except the very first
needed=$((total - 1))

# Generate sequential dates between Jan 1 and Feb 28, 2025
dates=()
current="2025-01-01"
for ((i=0; i<$needed; i++)); do
  dates+=("$current")
  # Jump ahead 1–3 days randomly
  current=$(date -d "$current +$((RANDOM % 3 + 1)) days" +%F)
  # Stop at Feb 28
  if [[ $(date -d "$current" +%s) -gt $(date -d "2025-02-28" +%s) ]]; then
    current="2025-02-28"
  fi
done

# Make sure we have enough dates
if [ ${#dates[@]} -lt $needed ]; then
  echo "❌ Not enough dates generated"
  exit 1
fi

echo "📅 Rewriting $needed commits with new sequential dates between Jan–Feb 2025..."

# Rewrite commit history
index=0
git filter-branch -f --env-filter "
if [ \$GIT_COMMIT != $first_commit ]; then
  export GIT_AUTHOR_DATE='${dates[$index]}T12:00:00'
  export GIT_COMMITTER_DATE='${dates[$index]}T12:00:00'
  index=$((index+1))
fi
" -- --all

echo "✅ Done!"
echo "Run: git log --pretty=format:'%h %ad %s' --date=short"
