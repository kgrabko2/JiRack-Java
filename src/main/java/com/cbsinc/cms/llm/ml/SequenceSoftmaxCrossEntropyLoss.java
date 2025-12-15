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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;

public class SequenceSoftmaxCrossEntropyLoss extends Loss {

    private final SoftmaxCrossEntropyLoss baseLoss;

    public SequenceSoftmaxCrossEntropyLoss() {
        super("SequenceSoftmaxCrossEntropyLoss");
        this.baseLoss = new SoftmaxCrossEntropyLoss("base", 1.0f, -1, true, true);
    }

    @Override
    public NDArray evaluate(NDList labels, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        NDArray lab = labels.singletonOrThrow();

        try (NDManager child = pred.getManager().newSubManager()) {
            long B = pred.getShape().get(0);
            long T = pred.getShape().get(1);
            long V = pred.getShape().get(2);

            NDArray flatPred = pred.reshape(new Shape(-1, V));
            NDArray flatLab = lab.reshape(new Shape(-1));

            return baseLoss.evaluate(new NDList(flatLab), new NDList(flatPred)); // ← скаляр
        }
    }
}