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