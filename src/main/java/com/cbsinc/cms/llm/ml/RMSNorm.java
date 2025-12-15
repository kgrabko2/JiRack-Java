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
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;

public class RMSNorm extends AbstractBlock {

    private static final float EPS = 1e-6f;

    private final Parameter weight;

    public RMSNorm(int embedDim) {
        weight = addParameter(Parameter.builder()
                .setName("weight")
                .setType(Parameter.Type.WEIGHT)
                .build());

        setInitializer(new ConstantInitializer(1.0f), Parameter.Type.WEIGHT);
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        NDArray x = inputs.get(0);
        Device device = x.getDevice();
        if (device == null) device = Device.cpu();

        NDArray w = ps.getValue(weight, device, training);

        NDArray rms = x.mul(x)
                       .mean(new int[]{-1}, true)
                       .add(EPS)
                       .sqrt();

        return new NDList(x.div(rms).mul(w));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] in) {
        return in;
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        long embedDim = inputShapes[0].get(inputShapes[0].dimension() - 1);
        weight.setShape(new Shape(embedDim));
        weight.initialize(manager, dataType);
    }
    
    
    public static void main(String[] args) {
    	try (NDManager mgr = NDManager.newBaseManager()) {
    	    RMSNorm norm = new RMSNorm(64);
    	    norm.initialize(mgr, DataType.FLOAT32, new Shape(1, 10, 64));

    	    System.out.println(norm.weight.getArray().sum()); // 64.0 (all 1.0)
    	}
		
	}
}