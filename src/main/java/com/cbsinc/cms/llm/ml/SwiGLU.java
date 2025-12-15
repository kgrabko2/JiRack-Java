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
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.nn.*;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

public class SwiGLU extends AbstractBlock {
    private final Parameter gateProj, upProj, downProj;
    private final int hiddenDim, intermediateDim;

    public SwiGLU(int hiddenDim, int intermediateDim) {
        if (hiddenDim <= 0 || intermediateDim <= 0)
            throw new IllegalArgumentException("Dimensions must be positive");

        this.hiddenDim = hiddenDim;
        this.intermediateDim = intermediateDim;

        gateProj = addParameter(Parameter.builder()
                .setName("gate_proj.weight")
                .setType(Parameter.Type.WEIGHT)
                .build());
        upProj = addParameter(Parameter.builder()
                .setName("up_proj.weight")
                .setType(Parameter.Type.WEIGHT)
                .build());
        downProj = addParameter(Parameter.builder()
                .setName("down_proj.weight")
                .setType(Parameter.Type.WEIGHT)
                .build());

        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.get(0);
        Shape original = x.getShape();

        Device device = x.getDevice();
        if (device == null) device = Device.cpu();

        NDArray xFlat = x.reshape(-1, hiddenDim);

        NDArray gateW = ps.getValue(gateProj, device, training);
        NDArray upW   = ps.getValue(upProj,   device, training);
        NDArray downW = ps.getValue(downProj, device, training);

        NDArray gate = xFlat.matMul(gateW);
        NDArray up   = xFlat.matMul(upW);

        // Manual SiLU: x * (1 / (1 + exp(-x)))
        NDArray expNeg = gate.neg().exp();           // exp(-gate)
        NDArray denom  = expNeg.add(1);              // 1 + exp(-gate)
        NDArray sigmoid = denom.div(1.0f);           // 1 / (1 + exp(-gate))
        NDArray act = gate.mul(sigmoid).mul(up);

        NDArray outFlat = act.matMul(downW);
        return new NDList(outFlat.reshape(original));
    }
    
    @Override
    public Shape[] getOutputShapes(Shape[] in) {
        return in;
    }

    /* -------------------------------------------------------------
       DJL 0.35.0 (and all older versions) only have
       Parameter.initialize(NDManager, DataType)
       ------------------------------------------------------------- */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        long inputDim = inputShapes[0].get(inputShapes[0].dimension() - 1);
        if (inputDim != hiddenDim) throw new IllegalArgumentException("Dim mismatch");

        gateProj.setShape(new Shape(hiddenDim, intermediateDim));
        upProj.setShape(new Shape(hiddenDim, intermediateDim));
        downProj.setShape(new Shape(intermediateDim, hiddenDim));

        gateProj.initialize(manager, dataType);
        upProj.initialize(manager, dataType);
        downProj.initialize(manager, dataType);
    }
    
    public static void main(String[] args) {
        try (NDManager mgr = NDManager.newBaseManager()) {
            SwiGLU glu = new SwiGLU(64, 256);
            glu.initialize(mgr, DataType.FLOAT32, new Shape(1, 10, 64));

            NDArray x = mgr.randomNormal(new Shape(1, 10, 64));
            NDArray y = glu.forward(new ParameterStore(mgr, false), new NDList(x), false)
                           .singletonOrThrow();

            System.out.println("Input:  " + x.getShape());
            System.out.println("Output: " + y.getShape());  // [1, 10, 64]
        }
    }
}