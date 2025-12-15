/**
# Copyright (c) 2025 CMS Manhattan
# All rights reserved.
# Author: Konstantin Vladimirovich Grabko
# Email: grabko@cmsmanhattan.com
# Phone: +1(516)777-0945
#
# MIT License
#
# Copyright (c) 2025 Konstantin Grabko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
*/

package com.cbsinc.cms.llm.ml;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Arrays;

public class GPTllm {

	public static final int VOCAB_SIZE = 50257;
	public static final int MODEL_DIM = 768;     // УМЕНЬШЕНО
	public static final int NUM_HEADS = 12;      // УМЕНЬШЕНО
	public static final int NUM_LAYERS = 6;
	public static final int MAX_SEQ_LEN = 8192;
	public static final int FFN_HIDDEN_DIM = 4 * MODEL_DIM;

    private static final Path MODEL_DIR = Paths.get("models");
    private static final Device DEVICE = Device.cpu();

    /* --------------------------------------------------------------- */

    public static Block buildModel() {
        SequentialBlock model = new SequentialBlock();

        model.add(new CustomEmbedding(VOCAB_SIZE, MODEL_DIM));
        // >>> ДОБАВЛЕНИЕ ЗАМОРОЖЕННОГО СЛОЯ С МЕТКОЙ <<<
        // Этот слой содержит вашу уникальную, неперезаписываемую метку.
        model.add(new FrozenSignatureLayer()); 
        model.add(new LearnedPositionalEmbedding(MAX_SEQ_LEN, MODEL_DIM));

        for (int i = 0; i < NUM_LAYERS; i++) {
            model.add(new TransformerBlock(i));
        }

        model.add(LayerNorm.builder().axis(2).build());

        model.add(Linear.builder()
                .setUnits(VOCAB_SIZE)
                .optBias(false)
                .build());

        return model;
    }

    /* ----------------------- LearnedPositionalEmbedding ----------------------- */
    static class LearnedPositionalEmbedding extends AbstractBlock {
        private static final byte VERSION = 1;
        private final int maxSeqLen, modelDim;
        public final Parameter posEmbedding;

        public LearnedPositionalEmbedding(int maxSeqLen, int modelDim) {
            super(VERSION);
            this.maxSeqLen = maxSeqLen;
            this.modelDim = modelDim;

            posEmbedding = addParameter(Parameter.builder()
                    .setName("position_embedding_weight")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(new Shape(maxSeqLen, modelDim))
                    .build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                         PairList<String, Object> params) {
            NDArray x = inputs.get(0);
            NDManager m = x.getManager();
            long B = x.getShape().get(0);
            long T = x.getShape().get(1);

            NDArray posWeight = ps.getValue(posEmbedding, m.getDevice(), training);
            NDArray pos = posWeight.get(new NDIndex("0:{}, :", T))
                    .expandDims(0).broadcast(B, T, modelDim);

            return new NDList(x.add(pos));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return inputShapes;
        }
    }

    /* --------------------------- TransformerBlock --------------------------- */
    static class TransformerBlock extends AbstractBlock {
        private static final byte VERSION = 1;
        private final MultiHeadAttention attn;
        private final SequentialBlock ffn;
        private final LayerNorm norm1, norm2;
        private final int layerIdx;

        public TransformerBlock(int layerIdx) {
            super(VERSION);
            this.layerIdx = layerIdx;
            attn = addChildBlock("attn", new MultiHeadAttention());

            norm1 = addChildBlock("norm1", LayerNorm.builder().axis(2).build());
            norm2 = addChildBlock("norm2", LayerNorm.builder().axis(2).build());

            ffn = addChildBlock("ffn",
                    new SequentialBlock()
                            .add(Linear.builder().setUnits(FFN_HIDDEN_DIM).optBias(false).build())
                            .add(Activation::gelu)
                            .add(Linear.builder().setUnits(MODEL_DIM).optBias(false).build()));
        }

        @Override
        public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape hidden = inputShapes[0];
            attn.initialize(manager, dataType, hidden);
            norm1.initialize(manager, dataType, hidden);
            norm2.initialize(manager, dataType, hidden);
            ffn.initialize(manager, dataType, hidden);
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                         PairList<String, Object> params) {
            NDArray x = inputs.get(0);
            NDArray pastKey = inputs.size() > 1 ? inputs.get(1) : null;
            NDArray pastValue = inputs.size() > 2 ? inputs.get(2) : null;

            NDList attnResult = attn.forward(ps,
                    pastKey == null ? new NDList(x) : new NDList(x, pastKey, pastValue),
                    training);
            NDArray attnOut = attnResult.get(0);

            x = norm1.forward(ps, new NDList(x.add(attnOut)), training).singletonOrThrow();
            NDArray ffnOut = ffn.forward(ps, new NDList(x), training).singletonOrThrow();
            x = norm2.forward(ps, new NDList(x.add(ffnOut)), training).singletonOrThrow();

            return new NDList(x);
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return new Shape[]{inputShapes[0]};
        }
    }

    /* -------------------------- MultiHeadAttention -------------------------- */
    static class MultiHeadAttention extends AbstractBlock {
        private static final byte VERSION = 1;
        private final Linear q, k, v, out;
        private final int h = NUM_HEADS;
        private final int d = MODEL_DIM / NUM_HEADS;
        private final float SCALE;

        public MultiHeadAttention() {
            super(VERSION);
            this.SCALE = (float) (1.0 / Math.sqrt(d));
            q = addChildBlock("q", Linear.builder().setUnits(MODEL_DIM).optBias(false).build());
            k = addChildBlock("k", Linear.builder().setUnits(MODEL_DIM).optBias(false).build());
            v = addChildBlock("v", Linear.builder().setUnits(MODEL_DIM).optBias(false).build());
            out = addChildBlock("out", Linear.builder().setUnits(MODEL_DIM).optBias(false).build());
        }

        @Override
        public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape hidden = inputShapes[0];
            q.initialize(manager, dataType, hidden);
            k.initialize(manager, dataType, hidden);
            v.initialize(manager, dataType, hidden);
            out.initialize(manager, dataType, hidden);
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                         PairList<String, Object> params) {
            NDArray x = inputs.get(0);
            NDManager m = x.getManager();
            long B = x.getShape().get(0), T = x.getShape().get(1);

            NDArray pastKey = inputs.size() > 1 ? inputs.get(1) : null;
            NDArray pastValue = inputs.size() > 2 ? inputs.get(2) : null;

            NDArray Q = q.forward(ps, new NDList(x), training).singletonOrThrow()
                    .reshape(B, T, h, d).transpose(0, 2, 1, 3);
            NDArray K = k.forward(ps, new NDList(x), training).singletonOrThrow()
                    .reshape(B, T, h, d).transpose(0, 2, 1, 3);
            NDArray V = v.forward(ps, new NDList(x), training).singletonOrThrow()
                    .reshape(B, T, h, d).transpose(0, 2, 1, 3);

            if (pastKey != null && pastValue != null) {
                K = NDArrays.concat(new NDList(pastKey, K), 2);
                V = NDArrays.concat(new NDList(pastValue, V), 2);
            }

            long T_total = K.getShape().get(2);

            NDArray scores = Q.matMul(K.transpose(0, 1, 3, 2)).mul(SCALE);

            if (pastKey == null) {
                NDArray mask = m.arange(T_total).reshape(-1, 1)
                        .lt(m.arange(T_total).reshape(1, -1))
                        .reshape(1, 1, T_total, T_total)
                        .broadcast(B, h, T_total, T_total);

                scores = NDArrays.where(mask,
                        m.full(scores.getShape(), Float.NEGATIVE_INFINITY), scores);
            }

            NDArray attn = scores.softmax(-1).matMul(V)
                    .transpose(0, 2, 1, 3).reshape(B, T, MODEL_DIM);

            NDArray output = out.forward(ps, new NDList(attn), training).singletonOrThrow();

            return new NDList(output, K, V);
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            Shape s = inputShapes[0];
            long B = s.get(0), T = s.get(1);
            Shape cache = new Shape(B, NUM_HEADS, -1, MODEL_DIM / NUM_HEADS);
            return new Shape[]{s, cache, cache};
        }
    }

    /* -------------------------- CustomEmbedding -------------------------- */
    static class CustomEmbedding extends AbstractBlock {
        private static final byte VERSION = 1;
        private final Parameter weight;
        private final int vocabSize;
        private final int embedDim;

        public CustomEmbedding(int vocabSize, int embedDim) {
            super(VERSION);
            this.vocabSize = vocabSize;
            this.embedDim = embedDim;

            weight = addParameter(Parameter.builder()
                    .setName("embedding_weight")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(new Shape(vocabSize, embedDim))
                    .build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                         PairList<String, Object> params) {
            NDArray ids = inputs.get(0);
            NDArray w = ps.getValue(weight, ids.getDevice(), training);
            return new NDList(w.get(ids));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] in) {
            Shape input = in[0];
            return new Shape[]{new Shape(input.get(0), input.get(1), embedDim)};
        }
    }
    
    /* -------------------------- FrozenSignatureLayer (Ваша метка) -------------------------- */
    static class FrozenSignatureLayer extends AbstractBlock {
        private static final byte VERSION = 1;
        private final Parameter signatureWeight;

        public FrozenSignatureLayer() {
            super(VERSION);

            // 1. Создание параметра с уникальным именем (Ваша метка)
            signatureWeight = addParameter(Parameter.builder()
                    .setName("H12_L6_D756_V50257_ML8192_cmsmanhattan_model_signature") 
                    .setType(Parameter.Type.WEIGHT)
                    // Маленький фиктивный вес, чтобы его было видно
                    .optShape(new Shape(1, 1)) 
                    .build());
            
            // 2. Установка Инициализатора в КОНСТРУКТОРЕ.
            // Это гарантирует, что при вызове initialize() на родительском блоке, 
            // этот параметр будет заполнен нулями.
            setInitializer(new ZeroInitializer(), Parameter.Type.WEIGHT); 
        }

        @Override
        public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            // 3. Замораживаем вес. 
            // К моменту вызова этого метода, weight уже инициализирован (создан и заполнен нулями).
            // Мы вызываем freeze(true) (то же, что и setRequiresGrad(false)), чтобы он не обучался.
            signatureWeight.freeze(true);
            
            // Если вы хотите явно проверить инициализацию, можно добавить эту строку, 
            // но она не нужна для корректной работы:
            // NDArray weight = signatureWeight.getArray(); 
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                         PairList<String, Object> params) {
            // Слой просто передает входные данные дальше, не меняя их.
            return inputs; 
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            // Входная и выходная форма совпадают
            return inputShapes;
        }
    }

    static class ZeroInitializer implements Initializer {
        @Override
        public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
            return manager.zeros(shape, dataType);
        }
    }

    public static class KVCache {
        NDArray key, value;
        public KVCache(NDArray k, NDArray v) { this.key = k; this.value = v; }
    }

    /* ------------------- СОЗДАНИЕ МОДЕЛИ ------------------- */
    public static void main(String[] args) throws Exception {
        System.out.println("CREATING EMPTY MODEL...");

        Model model = Model.newInstance("gpt", Device.cpu());
        Block block = buildModel();
        model.setBlock(block);

        // ПРАВИЛЬНЫЙ СПОСОБ
        block.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        block.setInitializer(new ZeroInitializer(), Parameter.Type.BIAS);

        try (NDManager manager = NDManager.newBaseManager()) {
            block.initialize(manager, DataType.FLOAT32, new Shape(1, 128, MODEL_DIM));
            model.save(Paths.get("models"), "gpt");
            System.out.println("MODEL CTEATED: models/gpt.params");
        }
    }

    /* -------------------------- Генерация -------------------------- */
    public static void generateExample(ai.djl.huggingface.tokenizers.HuggingFaceTokenizer tokenizer)
            throws IOException, MalformedModelException {
        // ... (оставляем как есть)
    }

    public static long sample(NDArray logits, float temperature) {
        NDManager m = logits.getManager();
        if (temperature <= 0) {
            return logits.argMax(-1).getLong();
        }

        NDArray probs = logits.sub(logits.max(new int[]{-1}, true)).exp();
        probs = probs.div(probs.sum(new int[]{-1}, true));

        NDArray cum = cumsum(probs.expandDims(0).expandDims(0));
        NDArray rand = m.randomUniform(0, 1, new Shape(1, 1, 1), DataType.FLOAT32)
                .broadcast(1, 1, VOCAB_SIZE);

        return rand.lt(cum).toType(DataType.FLOAT32, true).argMax(-1).getLong();
    }

    private static NDArray cumsum(NDArray array) {
        NDManager m = array.getManager();
        long V = array.getShape().get(2);
        NDArray result = m.zeros(array.getShape(), array.getDataType());
        NDArray prev = m.zeros(new Shape(1, 1), array.getDataType());

        for (long i = 0; i < V; i++) {
            NDArray slice = array.get("..., " + i);
            NDArray sum = i == 0 ? slice : prev.add(slice);
            result.set(new NDIndex("..., " + i), sum);
            prev = sum;
        }
        return result;
    }

    /* ------------------- ЗАГРУЗКА МОДЕЛИ ------------------- */
    private static Model loadModel() throws IOException, MalformedModelException {
        Model model = Model.newInstance("gpt", DEVICE);
        Block block = buildModel();
        model.setBlock(block);

        Path paramsFile = MODEL_DIR.resolve("gpt.params");
        if (Files.exists(paramsFile)) {
            model.load(MODEL_DIR, "gpt", Map.of("flavor", "djl"));
            System.out.println("Loaded weights from " + paramsFile);
        } else {
            System.out.println("No weights found – initializing randomly.");
            try (NDManager manager = NDManager.newBaseManager()) {
                block.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
                block.setInitializer(new ZeroInitializer(), Parameter.Type.BIAS);
                block.initialize(manager, DataType.FLOAT32, new Shape(1, 128, MODEL_DIM));
            }
        }
        return model;
    }
}