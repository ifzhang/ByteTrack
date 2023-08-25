import os

def rename_txt_files(output_folder):
    # Traverse the ByteTrack output directory
    for subdir in os.listdir(output_folder):
        subdir_path = os.path.join(output_folder, subdir)
        
        # Ensure it's a directory
        if os.path.isdir(subdir_path):
            # Get the BDD video name from the .mov file
            mov_files = [f for f in os.listdir(subdir_path) if f.endswith('.mov')]
            if mov_files:
                bdd_video_name = mov_files[0].replace('.mov', '')
                
                # Construct old and new txt file paths
                old_txt_path = os.path.join(output_folder, f"{subdir}.txt")
                new_txt_path = os.path.join(output_folder, f"{bdd_video_name}.txt")
                
                # Rename the txt file
                if os.path.exists(old_txt_path):
                    os.rename(old_txt_path, new_txt_path)
                    print(f"Renamed {old_txt_path} to {new_txt_path}")
                else:
                    print(f"File {old_txt_path} does not exist!")

# Example usage
output_folder = 'YOLOX_outputs/yolox_s_mix_det/track_vis'
rename_txt_files(output_folder)
