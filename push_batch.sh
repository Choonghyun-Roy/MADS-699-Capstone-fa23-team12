#!/bin/bash

# Navigate to the project directory
cd webapp/music_list

# Initialize a counter
count=0

# Batch size
batch_size=50

# A temporary file to hold the list of files to commit
batch_file=$(mktemp)

# Loop over all the mp3 files
for file in *.mp3; do
  # Add file to the batch list
  echo "$file" >> "$batch_file"
  
  # Increment the counter
  ((count++))
  
  # When we hit the batch size, commit and reset
  if ((count % batch_size == 0)); then
    # Stage all files listed in the batch file
    git add $(<"$batch_file")
    
    # Commit the batch
    git commit -m "Add batch of $batch_size music files"
    
    # Push the commit to the remote repository
    git push origin master
    
    # Clear the batch file for the next round
    > "$batch_file"
  fi
done

# If there are any leftover files after the loop, commit those too
if [[ -s "$batch_file" ]]; then
  git add $(<"$batch_file")
  git commit -m "Add final batch of music files"
  git push origin master
fi

# Remove the temporary file
rm "$batch_file"
