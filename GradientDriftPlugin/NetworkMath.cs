using System;

namespace GradientDriftPlugin;

// One pass through a policy network (drift_rl/policy.py MLP.forward): tanh hidden layers, a linear
// last one. weights[l] is row-major [outputs x inputs]; activations[0] holds the scaled inputs and
// activations[l + 1] receives layer l's outputs. Returns the last layer's buffer.
internal static class NetworkMath
{
    public static float[] Forward(float[][] weights, float[][] biases, float[][] activations)
    {
        int lastLayer = weights.Length - 1;
        for (int layer = 0; layer <= lastLayer; layer++)
        {
            float[] layerWeights = weights[layer];
            float[] layerBias = biases[layer];
            float[] layerInput = activations[layer];
            float[] layerOutput = activations[layer + 1];
            int inputCount = layerInput.Length;
            for (int o = 0; o < layerOutput.Length; o++)
            {
                float sum = layerBias[o];
                int rowStart = o * inputCount;
                for (int i = 0; i < inputCount; i++)
                {
                    sum += layerWeights[rowStart + i] * layerInput[i];
                }
                layerOutput[o] = layer < lastLayer ? MathF.Tanh(sum) : sum;
            }
        }
        return activations[lastLayer + 1];
    }
}
