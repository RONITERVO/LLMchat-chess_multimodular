import os
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES  # Provides drag & drop support

class PDFExtractorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Content Extractor")
        self.geometry("800x600")
        
        # Variables to store extracted content
        self.extracted_pdf_text = ""
        self.extracted_pdf_images = []
        self.full_pdf_content = {}
        
        # Create a frame that acts as the drop target.
        drop_frame = tk.Frame(self, bg="lightgray")
        drop_frame.pack(fill="both", expand=True)
        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', self.handle_pdf_drop)
        
        # Create a top-level frame for buttons.
        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", pady=5)
        
        open_pdf_btn = tk.Button(button_frame, text="Open PDF (Text)", command=self.open_pdf_file)
        open_pdf_btn.pack(side="left", padx=10)
        
        show_context_btn = tk.Button(button_frame, text="Show Text", command=self.show_context_window)
        show_context_btn.pack(side="left", padx=10)
        
        extract_images_btn = tk.Button(button_frame, text="Extract Images", command=self.open_pdf_images)
        extract_images_btn.pack(side="left", padx=10)
        
        extract_full_btn = tk.Button(button_frame, text="Extract Full Content", command=self.open_full_pdf_content)
        extract_full_btn.pack(side="left", padx=10)
    
    def open_pdf_file(self):
        """Open a PDF file using a file dialog and extract its text."""
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not pdf_path:
            return
        self.extracted_pdf_text = self.extract_text_from_pdf(pdf_path)
        messagebox.showinfo("PDF Loaded", "PDF text was successfully extracted!")
    
    def handle_pdf_drop(self, event):
        """
        When a PDF is dropped, automatically create an output folder (in the same directory
        as the PDF) and extract the full content into that folder.
        """
        files = self.tk.splitlist(event.data)
        for file in files:
            if file.lower().endswith(".pdf"):
                # Create output folder based on PDF name
                base_name = os.path.splitext(os.path.basename(file))[0]
                output_folder = os.path.join(os.path.dirname(file), f"extracted_{base_name}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                full_content = self.extract_full_content_from_pdf(file, output_folder)
                if full_content:
                    self.full_pdf_content = full_content
                break  # Process only the first PDF file dropped.
    
    def show_context_window(self):
        """Display a popup window with the extracted PDF text."""
        if not self.extracted_pdf_text:
            messagebox.showwarning("No PDF Loaded", "No PDF text available. Please load a PDF first.")
            return
        context_window = tk.Toplevel(self)
        context_window.title("Extracted PDF Text")
        text_widget = tk.Text(context_window, wrap="word", padx=10, pady=10)
        text_widget.insert("1.0", self.extracted_pdf_text)
        text_widget.config(state="disabled")
        text_widget.pack(fill="both", expand=True)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from all pages of the given PDF using PyMuPDF."""
        text_content = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract PDF text:\n{e}")
            return ""
        return text_content
    
    def open_pdf_images(self):
        """
        Open a PDF file using a file dialog, prompt the user to select an output folder,
        and then extract images from the PDF.
        """
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File for Image Extraction",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not pdf_path:
            return
        output_folder = filedialog.askdirectory(
            title="Select Output Folder for Extracted Images"
        )
        if not output_folder:
            messagebox.showwarning("No Folder Selected", "No output folder selected. Aborting.")
            return
        images = self.extract_images_from_pdf(pdf_path, output_folder)
        self.extracted_pdf_images = images
    
    def extract_images_from_pdf(self, pdf_path, output_folder):
        """
        Extract images from all pages of the given PDF.
        If saving in the native colorspace fails, attempt to convert to RGB.
        Returns a list of file paths for the saved images.
        """
        images_extracted = []
        try:
            doc = fitz.open(pdf_path)
            for page_number in range(len(doc)):
                page = doc[page_number]
                image_list = page.get_images(full=True)
                if image_list:
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        image_filename = os.path.join(output_folder, f"image_page{page_number+1}_{img_index}.png")
                        try:
                            pix.save(image_filename)
                        except Exception as e:
                            # If saving fails, attempt to convert to RGB colorspace
                            try:
                                pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                                pix_converted.save(image_filename)
                                pix_converted = None
                            except Exception as e2:
                                messagebox.showerror("Error", 
                                    f"Failed to save image (page {page_number+1}, index {img_index}): {e2}")
                        images_extracted.append(image_filename)
                        pix = None
            doc.close()
            messagebox.showinfo("Images Extracted", 
                f"Extracted {len(images_extracted)} images to:\n{output_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract images:\n{e}")
            return []
        return images_extracted
    
    def open_full_pdf_content(self):
        """
        Open a PDF file and prompt the user to select an output folder.
        Then extract the full content (metadata, text, images, annotations) into that folder.
        """
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File for Full Content Extraction",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not pdf_path:
            return
        output_folder = filedialog.askdirectory(
            title="Select Output Folder for Full Content Extraction"
        )
        if not output_folder:
            messagebox.showwarning("No Folder Selected", "No output folder selected. Aborting.")
            return
        full_content = self.extract_full_content_from_pdf(pdf_path, output_folder)
        if full_content:
            self.full_pdf_content = full_content
    
    def extract_full_content_from_pdf(self, pdf_path, output_folder):
        """
        Extract full content from the PDF including metadata, text, images, and annotations.
        If image saving fails due to an unsupported colorspace, it attempts an RGB conversion.
        Also saves text, metadata, and annotations as text files in the output folder.
        Returns a dictionary containing the extracted content.
        """
        full_content = {}
        try:
            doc = fitz.open(pdf_path)
            full_content['metadata'] = doc.metadata
            text_content = ""
            full_content['images'] = []
            full_content['annotations'] = []
            
            for page_number in range(len(doc)):
                page = doc[page_number]
                text_content += page.get_text() + "\n"
                
                # Extract images
                image_list = page.get_images(full=True)
                if image_list:
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        image_filename = os.path.join(output_folder, 
                                                      f"image_page{page_number+1}_{img_index}.png")
                        try:
                            pix.save(image_filename)
                        except Exception as e:
                            try:
                                pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                                pix_converted.save(image_filename)
                                pix_converted = None
                            except Exception as e2:
                                messagebox.showerror("Error", 
                                    f"Failed to save image (page {page_number+1}, index {img_index}): {e2}")
                        full_content['images'].append(image_filename)
                        pix = None
                
                # Extract annotations, if any.
                annot = page.first_annot
                while annot:
                    full_content['annotations'].append(annot.info)
                    annot = annot.next
            
            full_content['text'] = text_content
            doc.close()
            
            # Save extracted text.
            text_file = os.path.join(output_folder, "extracted_text.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text_content)
            
            # Save metadata.
            metadata_file = os.path.join(output_folder, "metadata.txt")
            with open(metadata_file, "w", encoding="utf-8") as f:
                for key, value in full_content['metadata'].items():
                    f.write(f"{key}: {value}\n")
            
            # Save annotations, if any.
            if full_content['annotations']:
                annotations_file = os.path.join(output_folder, "annotations.txt")
                with open(annotations_file, "w", encoding="utf-8") as f:
                    for annot in full_content['annotations']:
                        f.write(str(annot) + "\n")
            
            messagebox.showinfo("Full Content Extracted", 
                f"Full PDF content extracted successfully to:\n{output_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract full content:\n{e}")
            return None
        return full_content

if __name__ == "__main__":
    app = PDFExtractorApp()
    app.mainloop()

