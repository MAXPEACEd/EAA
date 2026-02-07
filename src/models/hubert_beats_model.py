import contextlib
import logging
import torch
import torch.nn as nn
from numba.tests.parfors_cache_usecases import self_run
from transformers import HubertModel, StoppingCriteriaList
from src.models.beats.BEATs import BEATsConfig, BEATs
from transformers import LlamaTokenizer, LlamaForCausalLM, StoppingCriteria, PreTrainedTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
import json
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=1):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

    def forward(self, x):
        # MultiheadAttention expects (seq_len, batch, hidden_size)
        x = x.permute(1, 0, 2)  # Convert to (seq_len, batch, hidden_size)
        attn_output, _ = self.self_attn(x, x, x)
        return attn_output.permute(1, 0, 2)  # Back to (batch, seq_len, hidden_size)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class AudioClassificationModel(nn.Module):
    def __init__(
        self,
        hubert_model_path=None,
        beats_model_path=None,
        llama_model_path=None,
        hidden_size=768,
        num_classes=7,
        freeze_hubert=True,
        freeze_beats=True,
        freeze_llama=True,
        prompt_path="/home/hongfei/emollm/prompts/train_prompt.json",
        prompt_template="USER: {}\nASSISTANT:",
        end_sym="</s>",
        max_txt_len=32,
        lora=True,
        lora_rank=2,
        lora_alpha=16,
        lora_dropout=0.2,
    ):
        """
        Initializes the model with Speech Encoder (HuBERT) and Acoustic Encoder (BEATs),
        and Cross-Attention for feature fusion.
        """
        super(AudioClassificationModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_prompt = False
        self.lora = lora
        self.end_sym = end_sym
        self.max_txt_len = max_txt_len
        # Speech Encoder: HuBERT
        if hubert_model_path:
            logging.info('Loading HuBERT Model')
            self.speech_encoder = HubertModel.from_pretrained(hubert_model_path)
            if freeze_hubert:
                for param in self.speech_encoder.parameters():
                    param.requires_grad = False
            # self.speech_encoder.eval()
            # Partially unfreeze the last layers of HuBERT
            for param in self.speech_encoder.encoder.layers[-2:].parameters():
                param.requires_grad = True
        else:
            self.speech_encoder = None

        # Acoustic Encoder: BEATs
        if beats_model_path:
            logging.info('Loading BEATs Model')
            beats_ckpt = torch.load(beats_model_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            self.acoustic_encoder = BEATs(beats_cfg)
            self.acoustic_encoder.load_state_dict(beats_ckpt['model'])
            if freeze_beats:
                for param in self.acoustic_encoder.parameters():
                    param.requires_grad = False
                    # Partially unfreeze the last layers of BEATs
                for param in self.acoustic_encoder.encoder.layers[-2:].parameters():
                    param.requires_grad = True
                # self.acoustic_encoder.eval()
        else:
            self.acoustic_encoder = None

        if llama_model_path:
            self.llama_tokenizer = PreTrainedTokenizerFast.from_pretrained(llama_model_path)
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # # New tokens to add (can be custom emotion words)
            # new_tokens = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
            #
            # # Check which tokens already exist to avoid duplicates
            # tokens_to_add = [tok for tok in new_tokens if tok not in self.llama_tokenizer.get_vocab()]

            # Add new tokens
            # self.llama_tokenizer.add_tokens(tokens_to_add)
            self.llama_tokenizer.padding_side = "right"

            # self.llama_tokenizer = LlamaTokenizer.from_pretrained("llama_model_path", use_fast=False)
            self.llama_model = LlamaForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.float16)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

            if freeze_llama:
                for param in self.llama_model.parameters():
                    param.requires_grad = False

        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')

        # Projection layers to align features to the same hidden size
        self.speech_proj = nn.Linear(self.speech_encoder.config.hidden_size, hidden_size) if self.speech_encoder else None
        self.acoustic_proj = nn.Linear(self.acoustic_encoder.cfg.encoder_embed_dim, hidden_size) if self.acoustic_encoder else None
        # self.fused_proj = nn.Linear(hidden_size * 4, self.llama_model.config.hidden_size)
        self.fused_proj = nn.Linear(hidden_size * 4, self.llama_model.config.hidden_size)
        # LayerNorm layers
        self.speech_ln = nn.LayerNorm(hidden_size) if self.speech_encoder else None
        self.acoustic_ln = nn.LayerNorm(hidden_size) if self.acoustic_encoder else None
        self.fused_ln = nn.LayerNorm(self.llama_model.config.hidden_size)  # Concatenation of 4 hidden_size features

        # Self-attention layers
        self.speech_self_attn = SelfAttention(768)
        self.acoustic_self_attn = SelfAttention(768)
        # Cross-Attention layer for feature fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.dropout = nn.Dropout(p=0.2)

        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),  # Simplified head
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # self.fc = nn.Linear(hidden_size,num_classes)
        self.feature_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Prepare prompts for a single task
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                logging.error("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))

            if "emotion_recognition" in raw_prompts:
                # filted_prompts = [raw_prompt for raw_prompt in raw_prompts["emotion_recognition"] if
                #                   "<SpeechHere>" in raw_prompt]
                single_prompt = raw_prompts["emotion_recognition"][0]
                # Format prompts
                self.prompt_dict["emotion_recognition"] = prompt_template.format(single_prompt)
                # self.prompt_dict["emotion_recognition"] = [prompt_template.format(p) for p in filted_prompts]
                # self.prompt = prompt_template.format(single_prompt)
            logging.info("Loading training prompts done!")

    def encode_audio(self, speech_input=None, acoustic_input=None, padding_mask=None):
        """
        Encodes the speech and acoustic inputs and returns fused features and attention mask.

        Returns:
            fused_features: Fused feature representation with LLaMA's hidden_size.
            audio_atts: Attention mask for fused features.
        """
        # Validate input
        if speech_input is None and acoustic_input is None:
            raise ValueError("At least one of speech_input or acoustic_input must be provided.")

        llama_hidden_size = self.llama_model.config.hidden_size  # Use LLaMA's hidden size

        # Process Speech Features (HuBERT)
        if self.speech_encoder and speech_input is not None:
            speech_features = self.speech_encoder(
                speech_input).last_hidden_state  # (batch, seq_len, speech_hidden_size)
            speech_features = self.speech_proj(speech_features)  # Project to hidden_size
            speech_features = self.speech_ln(speech_features)  # Apply LayerNorm
        else:
            speech_features = None

        # Process Acoustic Features (BEATs)
        if self.acoustic_encoder and acoustic_input is not None:
            acoustic_features, _ = self.acoustic_encoder.extract_features(acoustic_input, padding_mask=padding_mask,
                                                                          feature_only=True)  # (batch, seq_len, acoustic_hidden_size)
            acoustic_features = self.acoustic_proj(acoustic_features)  # Project to hidden_size
            acoustic_features = self.acoustic_ln(acoustic_features)  # Apply LayerNorm
        else:
            acoustic_features = None

        # Cross-attention fusion
        # if speech_features is not None and acoustic_features is not None:
        #     # Align sequence lengths
        #     max_len = max(speech_features.size(1), acoustic_features.size(1))
        #     speech_features = torch.nn.functional.pad(speech_features, (0, 0, 0, max_len - speech_features.size(1)))
        #     acoustic_features = torch.nn.functional.pad(acoustic_features,
        #                                                 (0, 0, 0, max_len - acoustic_features.size(1)))
        #
        #     # attended_speech, _ = self.cross_attention(
        #     #     query=speech_features.permute(1, 0, 2),  # Convert to (seq_len, batch, hidden_size)
        #     #     key=acoustic_features.permute(1, 0, 2),
        #     #     value=acoustic_features.permute(1, 0, 2)
        #     # )
        #     # attended_speech = attended_speech.permute(1, 0, 2)  # Back to (batch, seq_len, hidden_size)
        #
        #     attended_acoustic, _ = self.cross_attention(
        #         query=acoustic_features.permute(1, 0, 2),
        #         key=speech_features.permute(1, 0, 2),
        #         value=speech_features.permute(1, 0, 2)
        #     )
        #     attended_acoustic = attended_acoustic.permute(1, 0, 2)  # Back to (batch, seq_len, hidden_size)
        #
        #     # Concatenate the original and attended features
        #     # fused_features = torch.cat(
        #     #     [speech_features, acoustic_features, attended_speech, attended_acoustic],
        #     #     dim=-1  # Concatenating along the last dimension
        #     # )  # Shape: (batch, seq_len, 4 * hidden_size)
        #     # fused_features = torch.cat(
        #     #     [attended_speech, attended_acoustic, speech_features, acoustic_features],
        #     #     dim=-1  # Concatenating along the last dimension
        #     # )
        #
        #     fused_features = torch.cat([speech_features, attended_acoustic], dim=-1)

        # elif speech_features is not None:
        #     fused_features = speech_features  # Only speech
        # elif acoustic_features is not None:
        #     fused_features = acoustic_features  # Only acoustic
        # else:
        #     raise ValueError("No valid features from speech or acoustic inputs.")
            # Concatenate self-attention features

        # Get the maximum length
        max_len = max(speech_features.size(1), acoustic_features.size(1))

        # Pad to the same length
        speech_features = torch.nn.functional.pad(speech_features, (0, 0, 0, max_len - speech_features.size(1)))
        acoustic_features = torch.nn.functional.pad(acoustic_features, (0, 0, 0, max_len - acoustic_features.size(1)))

        # Apply self-attention
        att_speech = self.speech_self_attn(speech_features)
        att_acoustic = self.acoustic_self_attn(acoustic_features)
        if speech_features is not None and acoustic_features is not None:
            fused_features = torch.cat([speech_features, acoustic_features,att_speech,att_acoustic],
                                       dim=-1)  # (batch, seq_len, 2 * hidden_size)
        # Project fused features to LLaMA's hidden size
        fused_features = self.fused_proj(fused_features)
        fused_features = self.fused_ln(fused_features)  # LayerNorm

        # Generate attention mask
        audio_atts = torch.ones(fused_features.size()[:-1], dtype=torch.long, device=fused_features.device)

        # Debugging assertions
        assert fused_features.size(-1) == llama_hidden_size, \
            f"fused_features last dimension {fused_features.size(-1)} must match LLaMA hidden size {llama_hidden_size}"

        return fused_features, audio_atts

    def prompt_wrap(self, embeds, atts, prompts, multi_prompt=True):
        if not prompts:
            return embeds, atts

        if multi_prompt:
            # Prepare the before and after prompt lists
            p_before_list, p_after_list = [], []
            for prompt in prompts:
                b, a = prompt.split("<SpeechHere>")
                p_before_list.append(b)
                p_after_list.append(a)

            # Tokenize the before and after prompts
            p_before_tokens = self.llama_tokenizer(
                p_before_list, return_tensors="pt", add_special_tokens=False
            ).to(embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after_list, return_tensors="pt", add_special_tokens=False, padding="longest"
            ).to(embeds.device)

            # Get embeddings for before and after tokens
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

            # Initialize lists for wrapped embeddings and attention masks
            wrapped_embeds_list, wrapped_atts_list = [], []

            # Process each instance in the batch
            batch_size = embeds.size(0)
            for i in range(batch_size):
                for j in range(len(prompts)):
                    # Extract embeddings for a specific prompt
                    single_p_before_embeds = p_before_embeds[j].unsqueeze(0)  # (1, seq_len, hidden_size)
                    single_p_after_embeds = p_after_embeds[j].unsqueeze(0)  # (1, seq_len, hidden_size)
                    single_embeds = embeds[i].unsqueeze(0)  # (1, seq_len, hidden_size)

                    # Concatenate embeddings
                    wrapped_embeds = torch.cat([single_p_before_embeds, single_embeds, single_p_after_embeds], dim=1)

                    # Adjust attention masks
                    single_p_before_attn = p_before_tokens.attention_mask[j].unsqueeze(0)  # (1, seq_len)
                    single_p_after_attn = p_after_tokens.attention_mask[j].unsqueeze(0)  # (1, seq_len)
                    single_atts = atts[i].unsqueeze(0)  # (1, seq_len)
                    wrapped_atts = torch.cat([single_p_before_attn, single_atts, single_p_after_attn], dim=1)

                    # Append to lists
                    wrapped_embeds_list.append(wrapped_embeds)
                    wrapped_atts_list.append(wrapped_atts)

            # Stack results across all instances and prompts
            wrapped_embeds = torch.cat(wrapped_embeds_list,
                                       dim=0)  # (batch_size * len(prompts), new_seq_len, hidden_size)
            wrapped_atts = torch.cat(wrapped_atts_list, dim=0)  # (batch_size * len(prompts), new_seq_len)

            return wrapped_embeds, wrapped_atts
        else:
            batch_size = embeds.shape[0]
            p_before, p_after = prompts.split("<SpeechHere>")

            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False
            ).to(embeds.device)

            p_before_atts = p_before_tokens.attention_mask.expand(batch_size, -1)  # Expand to batch_size
            p_after_atts = p_after_tokens.attention_mask.expand(batch_size, -1)  # Expand to batch_size

            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1,
                                                                                                    -1) if not self.lora else self.llama_model.model.model.embed_tokens(
                p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1,
                                                                                                  -1) if not self.lora else self.llama_model.model.model.embed_tokens(
                p_after_tokens.input_ids).expand(batch_size, -1, -1)

            wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
            wrapped_atts = torch.cat([p_before_atts, atts, p_after_atts], dim=1)
        return wrapped_embeds, wrapped_atts

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples, speech_input=None, acoustic_input=None, padding_mask=None, verbose=True):
        """
        Forward method for speech and acoustic inputs.
        Args:
            samples: Input samples containing emotion and raw audio data.
            speech_input: Input features for the HuBERT model.
            acoustic_input: Input features for the BEATs model.
            padding_mask: Mask for acoustic input.

        Returns:
            A dictionary containing the loss.
        """

        # Compute audio features
        fused_features, audio_atts = self.encode_audio(
            speech_input=speech_input,
            acoustic_input=acoustic_input,
            padding_mask=padding_mask
        )
        fused_features = self.dropout(fused_features)

        # Get text features (utterance_with_context)
        context_tokens = self.llama_tokenizer(
            samples["utterance_with_context"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=128,
            add_special_tokens=False
        ).to(fused_features.device)

        context_embeddings = self.llama_model.model.model.embed_tokens(
            context_tokens.input_ids
        )

        # Build text attention mask
        context_attn_mask = torch.ones(
            (context_embeddings.size(0), context_embeddings.size(1)),  # [batch_size, context_length]
            dtype=torch.long,
            device=fused_features.device
        )

        # Target text (emotion label)
        texts = [t + self.llama_tokenizer.eos_token for t in samples["emotion"]]
        to_regress_tokens = self.llama_tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(fused_features.device)

        # Target text embeddings
        to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)

        # Build labels
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        # Create start token (BOS)
        bos = torch.ones(
            [fused_features.shape[0], 1],  # batch_size, 1
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # [batch_size, 1, hidden_dim]

        prompt = self.prompt_dict.get("emotion_recognition", None)
        if prompt:
            wrapped_embeds, wrapped_atts = self.prompt_wrap(
                fused_features, audio_atts, prompt, multi_prompt=self.multi_prompt
            )

            if wrapped_embeds is None or wrapped_embeds.numel() == 0:
                raise ValueError("wrapped_embeds is empty. Ensure valid prompts are provided.")


        # Ensure order: BOS -> Prompt + Audio -> Context -> Labels
        inputs_embeds = torch.cat([bos_embeds, wrapped_embeds, context_embeddings, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [torch.ones_like(bos_embeds[:, :, 0], dtype=torch.long), wrapped_atts, context_attn_mask,
             to_regress_tokens.attention_mask],
            dim=1
        )

        # Ensure targets align so LLaMA only predicts emotion labels
        empty_targets = torch.ones(
            [attention_mask.shape[0], attention_mask.shape[1] - targets.shape[1]],  # Align to attention mask length
            dtype=torch.long,
        ).to(fused_features.device).fill_(-100)

        targets = torch.cat([empty_targets, targets], dim=1)  # [batch_size, seq_len]


        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            logits = outputs.logits

        if verbose:
            nvocab = self.llama_model.config.vocab_size

            # Keep logits aligned to targets
            logits = outputs.logits[:, -targets.shape[1]:, :]

            # Get predictions
            results = logits.contiguous().view(-1, nvocab).argmax(dim=-1)

            # Ensure targets shape is correct
            labels = targets.contiguous().view(-1)

            # Build mask for valid tokens
            mask = (labels != -100)

            # Count correct predictions
            correct = (results[mask] == labels[mask]).float().sum()

            # Count total tokens
            total = mask.sum()

            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def evaluate_model(self, samples, speech_input, acoustic_input, padding_mask, save_path=None):
        pred_texts = self.generate(
            samples=samples,
            speech_input=speech_input,
            acoustic_input=acoustic_input,
            padding_mask=padding_mask
        )

        ground_truths = samples["emotion"]

        results = []
        for pred, gt in zip(pred_texts, ground_truths):
            cleaned_pred = pred.strip().replace("<|end_of_text|>", "")  # Remove </s> and trim spaces
            cleaned_gt = gt.strip().replace("<|end_of_text|>", "")  # Remove </s> and trim spaces
            match = cleaned_pred == cleaned_gt  # Compare cleaned prediction to ground truth

            # Log per-sample results
            logging.info(f"Prediction: {cleaned_pred}, Ground Truth: {cleaned_gt}, Match: {match}")

            results.append({
                "prediction": cleaned_pred,
                "ground_truth": cleaned_gt,
                "match": match
            })

        # Compute accuracy
        accuracy = sum(r["match"] for r in results) / len(results)
        logging.info(f"Classification Accuracy: {accuracy * 100:.2f}%")

        # Save results if a path is provided
        # if save_path:
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         json.dump(results, f, ensure_ascii=False, indent=4)
        #     logging.info(f"Results saved to {save_path}")

        return results  # Per-sample match results

    def generate(self, samples, speech_input=None, acoustic_input=None, padding_mask=None):
        batch_size = speech_input.shape[0]

        # ---------------- 1️⃣  Process audio and text separately ---------------- #

        # Compute audio features
        fused_features, audio_atts = self.encode_audio(
            speech_input=speech_input,
            acoustic_input=acoustic_input,
            padding_mask=padding_mask
        )
        fused_features = self.dropout(fused_features)

        # Get text features (utterance_with_context)
        context_tokens = self.llama_tokenizer(
            samples["utterance_with_context"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=128,
            add_special_tokens=False
        ).to(fused_features.device)

        context_embeddings = self.llama_model.model.model.embed_tokens(
            context_tokens.input_ids
        )

        # Build text attention mask
        context_attn_mask = torch.ones(
            (context_embeddings.size(0), context_embeddings.size(1)),  # [batch_size, context_length]
            dtype=torch.long,
            device=fused_features.device
        )

        # Create start token (BOS)
        bos = torch.ones(
            [batch_size, 1],  # batch_size, 1
            dtype=torch.long,
            device=fused_features.device
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # [batch_size, 1, hidden_dim]

        # ---------------- 2️⃣  Let LLaMA read the prompt ---------------- #

        prompt = self.prompt_dict.get("emotion_recognition", None)
        if prompt:
            wrapped_embeds, wrapped_atts = self.prompt_wrap(
                fused_features, audio_atts, prompt, multi_prompt=self.multi_prompt
            )

            if wrapped_embeds is None or wrapped_embeds.numel() == 0:
                raise ValueError("wrapped_embeds is empty. Ensure valid prompts are provided.")

        # ---------------- 3️⃣  Concatenate all information ---------------- #
        # Ensure order: BOS -> Prompt + Audio -> Context
        inputs_embeds = torch.cat([bos_embeds, wrapped_embeds, context_embeddings], dim=1)
        attention_mask = torch.cat(
            [torch.ones_like(bos_embeds[:, :, 0], dtype=torch.long), wrapped_atts, context_attn_mask],
            dim=1
        )

        # ---------------- 4️⃣  Generate text ---------------- #
        stop_words_ids = [torch.tensor([128001]).to(fused_features.device)]  # Stop token ID
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=4,  # Limit max generated tokens
                stopping_criteria=stopping_criteria,  # Stop conditions
                num_beams=1,  # Beam search
                do_sample=False,  # Disable sampling
                min_length=1,  # Minimum length
                top_p=1,  # Nucleus sampling
                repetition_penalty=1.1,  # Avoid repetition
                length_penalty=1,  # Control length
                attention_mask=attention_mask,  # Restrict attention to valid parts
                pad_token_id=self.llama_tokenizer.pad_token_id,
            )

            # Decode generated text and remove special tokens
            generated_texts = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts


def evaluate_accuracy(predictions, ground_truths):
    assert len(predictions) == len(ground_truths), "length not match"

    correct = sum([pred == true for pred, true in zip(predictions, ground_truths)])
    accuracy = correct / len(ground_truths)
    return accuracy