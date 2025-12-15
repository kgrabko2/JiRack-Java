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