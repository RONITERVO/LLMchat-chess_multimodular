# Copyright (C) <2025>  <Roni Sam Daniel Tervo>
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import fitz  # PyMuPDF
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

class PDFExtractorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Content Extractor")
        self.geometry("400x300")
        self.OUTPUT_FOLDER = "C:\\Users\\ronit\\OneDrive\\Tiedostot\\extracted_pdf_data"
        
        # Create drop area with better visual feedback
        drop_frame = tk.Frame(self, bg="#f0f0f0", borderwidth=2, relief="groove")
        drop_frame.pack(fill="both", expand=True, padx=20, pady=20)
        tk.Label(drop_frame, text="Drag & Drop PDF Here", 
                bg="#f0f0f0", fg="#404040", font=("Arial", 18, "bold")).pack(expand=True)
        
        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', self.handle_pdf_drop)
    
    def handle_pdf_drop(self, event):
        files = self.tk.splitlist(event.data)
        for file in files:
            if file.lower().endswith(".pdf"):
                os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
                self.extract_full_content_from_pdf(file, self.OUTPUT_FOLDER)
                break
    
    def show_auto_close_message(self, title, message):
        msg = tk.Toplevel(self)
        msg.title(title)
        msg.geometry("450x100")
        tk.Label(msg, text=message, wraplength=400, pady=20).pack(expand=True)
        self.after(3000, self.destroy)

    def safe_convert_pixmap(self, pix):
        """Convert pixmap to RGB if not in supported color space"""
        if pix.colorspace and pix.colorspace.n not in (fitz.csGRAY.n, fitz.csRGB.n):
            return fitz.Pixmap(fitz.csRGB, pix)
        return pix
    
    def extract_full_content_from_pdf(self, pdf_path, output_folder):
        try:
            doc = fitz.open(pdf_path)
            text_output = os.path.join(output_folder, "text")
            image_output = os.path.join(output_folder, "images")
            os.makedirs(text_output, exist_ok=True)
            os.makedirs(image_output, exist_ok=True)

            # Extract text and metadata
            text_content = "\n".join(page.get_text() for page in doc)
            with open(os.path.join(text_output, "full_text.txt"), "w", encoding="utf-8") as f:
                f.write(text_content)
            
            # Save metadata
            with open(os.path.join(text_output, "metadata.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(f"{k}: {v}" for k, v in doc.metadata.items()))

            # Improved image extraction with color space handling
            for page_number in range(len(doc)):
                page = doc[page_number]
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert problematic color spaces
                    pix = self.safe_convert_pixmap(pix)
                    
                    # Generate filename and save
                    filename = f"page{page_number+1}_img{img_index}.png"
                    image_path = os.path.join(image_output, filename)
                    
                    try:
                        pix.save(image_path)
                    except Exception as save_error:
                        error_msg = f"Failed to save image {filename}:\n{save_error}"
                        self.show_auto_close_message("Error", error_msg)
                        continue
                    finally:
                        pix = None  # Ensure Pixmap cleanup

            doc.close()
            self.show_auto_close_message("Success", f"All content extracted to:\n{output_folder}")
            
        except Exception as e:
            self.show_auto_close_message("Error", f"Extraction failed:\n{str(e)}")

if __name__ == "__main__":
    app = PDFExtractorApp()
    app.mainloop()
