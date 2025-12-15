/**
# Copyright (c) 2025 CMS Manhattan
# All rights reserved.
# Author: Konstantin Vladimirovich Grabko
# Email: grabko@cmsmanhattan.com
# Phone: +1(516)777-0945
#
# This file is part of a project authored by CMS Manhattan. You may use, distribute, and modify
# this code under the terms of the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
# Please read <http://www.gnu.org/licenses/>.
# JiRack on Java — final clean version, December 2025
*/


package com.cbsinc.cms.llm.ml;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GPTFineTuner {

    private static int TRAIN_SEQ_LEN = 64;
    private static String tokenizerName = "gpt2";
    private static String MODEL_NAME = "gpt";
    private static int BATCH_SIZE = 1;
    private static final String OUTPUT_DIR = "build/fine_tuning_output";
    private static Path MODEL_DIR = Paths.get("models");
    private static final Device DEVICE = Device.cpu();
    //private static final Device DEVICE = Device.gpu(); // Use GPU if available
    private static Path TRAIN_CSV = Paths.get("model-dir/datasets/dialogues_text.txt");
    private static int EPOCHS = 4;
    private static final int KEEP_LAST_EPOCHS = 3;

    private long totalSteps = 0;
    private NDManager datasetManager; // Persistent manager for dataset

    public TrainingConfig setupTrainingConfig() {
        Loss loss = new SequenceSoftmaxCrossEntropyLoss();
        Optimizer optimizer = Optimizer.adam()
                // REDUCED LEARNING RATE: Try 1e-5f or 5e-6f
                .optLearningRateTracker(Tracker.fixed(1e-5f)) 
                .optWeightDecays(0.01f)
                .optClipGrad(1.0f)
                .build();

        return new DefaultTrainingConfig(loss)
                .optOptimizer(optimizer)
                .addEvaluator(new Accuracy("Accuracy", -1))
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }

 // GPTFineTuner.java:loadModel() - CORRECTED
    private Model loadModel() throws IOException, MalformedModelException {
        Model model = Model.newInstance("gpt", DEVICE);
        Block block = GPTllm.buildModel();
        model.setBlock(block);

        Path paramsFile = MODEL_DIR.resolve("gpt-0000.params");
        if (Files.exists(paramsFile)) {
            model.load(MODEL_DIR, "gpt", Map.of("flavor", "djl"));
            System.out.println("Loaded weights from " + paramsFile);
        } else {
            System.out.println("No weights found – initializing randomly.");
            // **Initialization logic moved to train() after trainer is created.**
            // Only set the initializers here.
            block.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
            block.setInitializer(new GPTllm.ZeroInitializer(), Parameter.Type.BIAS);
        }
        return model;
    }

    public ArrayDataset loadDataset() throws IOException, TranslateException {
        if (!Files.exists(TRAIN_CSV)) {
            throw new IOException("Dataset not found: " + TRAIN_CSV.toAbsolutePath());
        }

        String text = Files.readString(TRAIN_CSV);
        System.out.println("Loaded " + text.length() + " characters (~" + (text.length() / 4) + " tokens)");

        // FIXED: Create persistent manager ONCE
        if (datasetManager == null) {
            datasetManager = NDManager.newBaseManager();
        }

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder()
                .optTokenizerName(tokenizerName)
                .optTruncation(false)
                .optPadding(false)
                .optMaxLength(Integer.MAX_VALUE)
                .build()) {

            Encoding encoding = tokenizer.encode(text);
            long[] ids = encoding.getIds();
            System.out.println("Total tokens: " + ids.length);

            if (ids.length <= TRAIN_SEQ_LEN + 1) {
                throw new IllegalArgumentException("Text too short for block size " + TRAIN_SEQ_LEN);
            }

            long nBlocks = (ids.length - 1) / TRAIN_SEQ_LEN;

            // ---- Build the two big arrays under the *persistent* manager ----
            NDArray full = datasetManager.create(ids).toType(DataType.INT64, false);
            NDArray inputs = datasetManager.zeros(new Shape(nBlocks, TRAIN_SEQ_LEN), DataType.INT64);
            NDArray labels = datasetManager.zeros(new Shape(nBlocks, TRAIN_SEQ_LEN), DataType.INT64);

         // FINAL CORRECTED CODE SECTION:
            for (long i = 0; i < nBlocks; i++) {
                long start = i * TRAIN_SEQ_LEN;

                // 1. Get a view of the input slice (no duplication)
                try (NDArray inSlice = full.get(start + ":" + (start + TRAIN_SEQ_LEN))
                                           .toType(DataType.INT64, false)) { 
                    // 2. Copy data from the view into the persistent 'inputs' array
                    inputs.set(new NDIndex(i + ", :"), inSlice);
                } // inSlice (the temporary view) is closed here.

                // 1. Get a view of the label slice (no duplication)
                try (NDArray lblSlice = full.get((start + 1) + ":" + (start + TRAIN_SEQ_LEN + 1))
                                            .toType(DataType.INT64, false)) {
                    // 2. Copy data from the view into the persistent 'labels' array
                    labels.set(new NDIndex(i + ", :"), lblSlice);
                } // lblSlice (the temporary view) is closed here.
            }

            // No detach() – the arrays stay attached to datasetManager
            full.close();   // we are done with the source array

            totalSteps = (nBlocks + BATCH_SIZE - 1) / BATCH_SIZE;
            System.out.println("Training on " + nBlocks + " sequences (" + totalSteps + " steps per epoch)");

            return new ArrayDataset.Builder()
                    .setData(inputs)
                    .optLabels(labels)
                    .setSampling(BATCH_SIZE, true)
                    .build();
        }
    }

    private void cleanupOldEpochs(Path baseDir, int keepLast) {
        try (Stream<Path> pathStream = Files.list(baseDir)) {
            List<Path> epochDirs = pathStream
                    .filter(p -> p.getFileName().toString().startsWith("epoch"))
                    .sorted(Comparator.comparingInt(p -> {
                        String name = p.getFileName().toString();
                        return Integer.parseInt(name.replace("epoch", ""));
                    }))
                    .collect(Collectors.toList());

            if (epochDirs.size() <= keepLast) return;

            for (int i = 0; i < epochDirs.size() - keepLast; i++) {
                Path dir = epochDirs.get(i);
                try (Stream<Path> files = Files.walk(dir)) {
                    files.sorted(Comparator.reverseOrder())
                         .map(Path::toFile)
                         .forEach(f -> {
                             if (f.delete()) {
                                 System.out.println("Deleted: " + f);
                             }
                         });
                } catch (IOException e) {
                    System.err.println("Failed to clean: " + dir);
                }
                System.out.println("Cleaned old epoch: " + dir.getFileName());
            }
        } catch (IOException e) {
            System.err.println("Failed to list epochs: " + e.getMessage());
        }
    }

    public void train() throws Exception {
        Model model = null;
        ArrayDataset dataset = null;

        try {
            model = loadModel();
            dataset = loadDataset();
            TrainingConfig config = setupTrainingConfig();

            try (Trainer trainer = model.newTrainer(config)) {
                // FIX: This call must happen here, after the Trainer is created, 
                // but before the training loop starts.
                // If the model was loaded from a file (isInitialized=true), 
                // this prepares the trainer's internal state.
                // If the model was NOT loaded (isInitialized=false), 
                // this calls block.initialize() using the trainer's NDManager, 
                // which keeps the native resources alive.
                trainer.initialize(new Shape(BATCH_SIZE, TRAIN_SEQ_LEN));
                
                System.out.println("\n--- STARTING FINE-TUNING ---");

                for (int epoch = 0; epoch < EPOCHS; epoch++) {
                    System.out.println("Epoch " + (epoch + 1) + "/" + EPOCHS);
                    int step = 0;

                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        try (batch) {
                            try (GradientCollector gc = trainer.newGradientCollector()) {
                                NDList data = batch.getData();
                                NDList labels = batch.getLabels();

                                NDList predictions = trainer.forward(data);
                                NDArray logits = predictions.singletonOrThrow();
                                NDArray loss = config.getLossFunction().evaluate(labels, new NDList(logits));
                                float lossValue = loss.getFloat();
                                float perplexity = (float) Math.exp(lossValue);

                                System.out.printf("[%d/%d] Step %d | Loss: %.3f | PPL: %.2f%n",
                                        step + 1, totalSteps, step, lossValue, perplexity);

                                gc.backward(loss);

                                // CLOSE NDArrays/NDLists that you won't use anymore
                                predictions.close();
                                logits.close();
                                loss.close();
                            }
                            trainer.step();
                        } catch (Exception e) {
                            System.err.println("Error at step " + step + ": " + e.getMessage());
                            e.printStackTrace();
                        }
                        step++;
                    }

                    Path epochDir = Paths.get(OUTPUT_DIR, "epoch" + (epoch + 1));
                    Files.createDirectories(epochDir);
                    model.save(epochDir, MODEL_NAME);
                    System.out.println("Epoch " + (epoch + 1) + " saved to: " + epochDir.toAbsolutePath());

                    cleanupOldEpochs(Paths.get(OUTPUT_DIR), KEEP_LAST_EPOCHS);
                    trainer.notifyListeners(l -> l.onEpoch(trainer));
                }
            } // Trainer is closed here.

            Path finalOut = Paths.get(OUTPUT_DIR, "final");
            Files.createDirectories(finalOut);
            model.save(finalOut, MODEL_NAME);
            System.out.println("Training complete. Final model saved to: " + finalOut.toAbsolutePath());

        } finally {
        	if (model != null) {
                model.close();
            }
            if (datasetManager != null) {
                datasetManager.close();
                System.out.println("Closed dataset NDManager.");
            }
        }
    }

    public static void main(String[] args) {
        System.out.println("java -cp Jirackkit.jar com.cbsinc.cms.llm.ml.GPTFineTuner [args]");
        System.setProperty("ai.djl.ndarray.debug", "true");

        if (args.length > 0) BATCH_SIZE = Integer.parseInt(args[0]);
        if (args.length > 1) TRAIN_SEQ_LEN = Integer.parseInt(args[1]);
        if (args.length > 2) EPOCHS = Integer.parseInt(args[2]);
        if (args.length > 3) MODEL_NAME = args[3];
        if (args.length > 4) MODEL_DIR = Paths.get(args[4]);
        if (args.length > 5) tokenizerName = args[5];
        if (args.length > 6) TRAIN_CSV = Paths.get(args[6]);

        System.out.println("Looking for dataset: " + TRAIN_CSV.toAbsolutePath());
        try {
            new GPTFineTuner().train();
        } catch (Exception e) {
            System.err.println("Training failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}