from transformers import AutoTokenizer, AutoModel, DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import torch
class DonutModel:
    def __init__(self, model_name):
        self.processor = DonutProcessor.from_pretrained("sccengizlrn/invoices-donut-model-v1")
        self.model = VisionEncoderDecoderModel.from_pretrained("sccengizlrn/invoices-donut-model-v1")

    def process_document(self, image):
        try:
            # Tokenizer ile encoder inputs hazırlama
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # Tokenizer ile decoder inputs hazırlama
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

            outputs = self.model.generate(pixel_values.to(device),
                                          decoder_input_ids=decoder_input_ids.to(device),
                                          max_length=self.model.decoder.config.max_position_embeddings,
                                          early_stopping=True,
                                          pad_token_id=self.processor.tokenizer.pad_token_id,
                                          eos_token_id=self.processor.tokenizer.eos_token_id,
                                          use_cache=True,
                                          num_beams=1,
                                          bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                          return_dict_in_generate=True,
                                          output_scores=True,)
            

            # Model ile cevap üretme
            outputs = self.model.generate(pixel_values.to(device),
                               decoder_input_ids=decoder_input_ids.to(device),
                               max_length=self.model.decoder.config.max_position_embeddings,
                               early_stopping=True,
                               pad_token_id=self.processor.tokenizer.pad_token_id,
                               eos_token_id=self.processor.tokenizer.eos_token_id,
                               use_cache=True,
                               num_beams=1,
                               bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                               return_dict_in_generate=True,
                               output_scores=True,)
            

            # Cevabı işleme
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # ilk görev başlangıç belirteci kaldırma
            
            return self.processor.token2json(sequence)

        except Exception as e:
            print("Hata:", str(e))
            return None
