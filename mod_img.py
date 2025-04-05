from PIL import Image
import os

def remove_faixas(input_folder, output_folder, top_remove, bottom_remove):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            
            width, height = image.size
            new_height = height - top_remove - bottom_remove
            
            if new_height > 0:
                cropped_image = image.crop((0, top_remove, width, height - bottom_remove))
                
                output_path = os.path.join(output_folder, filename)
                cropped_image.save(output_path)
                print(f"Imagem {filename} processada e salva em {output_path}")
            else:
                print(f"Erro: Faixas removidas são maiores que a altura da imagem {filename}")

input_folder = "IR"
output_folder = "output_folder"
top_remove = 50
bottom_remove = 50

remove_faixas(input_folder, output_folder, top_remove, bottom_remove)