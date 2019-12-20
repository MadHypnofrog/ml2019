package ml;

import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class CNN {

    enum Type {
        RELU, POOL, CNVM, CNVE, CNVC
    }

    static class Layer {
        Type type;
        double[][][] matrix;
        double[][][][] kernel;
        double[][][][] kernelDer;
        double[][][][] kernelNorms;
        int h, k, s, p;
        double alpha;
        double[][][] derivative;

        Layer() {
            this.alpha = 0.01;
            type = Type.RELU;
        }

        Layer(int s) {
            this.s = s;
            type = Type.POOL;
        }

        Layer(double[][][][] kernel, int h, int k, int s, int p, String border) {
            this.kernel = kernel;
            this.kernelNorms = new double[kernel.length][kernel[0].length][kernel[0][0].length][kernel[0][0][0].length];
            this.h = h;
            this.k = k;
            this.s = s;
            this.p = p;
            type = Type.valueOf(border);
        }

        double[][][] process(double[][][] matrix) {
            this.matrix = matrix;
            switch (type) {
                case RELU: {
                    double[][][] result = new double[matrix.length][matrix[0].length][matrix[0][0].length];
                    for (int i = 0; i < matrix.length; i++) {
                        for (int j = 0; j < matrix[0].length; j++) {
                            for (int k = 0; k < matrix[0][0].length; k++) {
                                result[i][j][k] = matrix[i][j][k] > 0 ? matrix[i][j][k] : matrix[i][j][k] * alpha;
                            }
                        }
                    }
                    return result;
                }
                case POOL: {
                    int len = matrix[0].length / s;
                    double[][][] result = new double[matrix.length][len][len];
                    for (int i = 0; i < matrix.length; i++) {
                        for (int j = 0; j < len; j++) {
                            for (int k = 0; k < len; k++) {
                                double max = matrix[i][s * j][s * k];
                                for (int x = s * j; x < s * (j + 1); x++) {
                                    for (int y = s * k; y < s * (k + 1); y++) {
                                        max = Math.max(max, matrix[i][x][y]);
                                    }
                                }
                                result[i][j][k] = max;
                            }
                        }
                    }
                    return result;
                }
                default: {
                    int newDim = (matrix[0].length + 2 * p - k) / s + 1;
                    double[][][] result = new double[h][newDim][newDim];
                    double[][][] extended = extend3D(matrix, p, type);
                    for (int i = 0; i < h; i++) {
                        for (int j = 0; j < newDim; j++) {
                            for (int c = 0; c < newDim; c++) {
                                double res = 0;
                                for (int z = 0; z < matrix.length; z++) {
                                    for (int x = s * j; x < s * j + k; x++) {
                                        for (int y = s * c; y < s * c + k; y++) {
                                            res += kernel[i][z][x - s * j][y - s * c]
                                                    * extended[z][x][y];
                                        }
                                    }
                                }
                                result[i][j][c] = res;
                            }
                        }
                    }
                    return result;
                }
            }

        }

        void calcDerivative(double[][][] prevD, double[][][] nextMatrix) {
            switch (type) {
                case RELU: {
                    derivative = new double[prevD.length][prevD[0].length][prevD[0].length];
                    for (int z = 0; z < prevD.length; z++) {
                        for (int j = 0; j < prevD[0].length; j++) {
                            for (int k = 0; k < prevD[0][0].length; k++) {
                                derivative[z][j][k] = matrix[z][j][k] >= 0 ? prevD[z][j][k] : prevD[z][j][k] * alpha;
                            }
                        }
                    }
                    break;
                }
                case POOL: {
                    derivative = new double[matrix.length][matrix[0].length]
                            [matrix[0].length];
                    for (int z = 0; z < nextMatrix.length; z++) {
                        for (int j = 0; j < nextMatrix[0].length; j++) {
                            for (int k = 0; k < nextMatrix[0][0].length; k++) {
                                double target = nextMatrix[z][j][k];
                                for (int x = s * j; x < s * (j + 1); x++) {
                                    for (int y = s * k; y < s * (k + 1); y++) {
                                        if (matrix[z][x][y] == target) {
                                            derivative[z][x][y] = prevD[z][j][k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                default: {
                    derivative = new double[matrix.length][matrix[0].length + 2 * p]
                            [matrix[0].length + 2 * p];
                    for (int h = 0; h < this.h; h++) {
                        for (int j = 0; j < prevD[h].length; j++) {
                            for (int k = 0; k < prevD[h].length; k++) {

                                for (int z = 0; z < matrix.length; z++) {
                                    for (int x = s * j; x < s * j + this.k; x++) {
                                        for (int y = s * k; y < s * k + this.k; y++) {
                                            derivative[z][x][y] += kernel[h][z][x - s * j][y - s * k]
                                                    * prevD[h][j][k];
                                        }
                                    }
                                }

                            }
                        }
                    }
                    derivative = fold3D(derivative, p, type);
                    double[][][] extended = extend3D(matrix, p, type);
                    kernelDer = new double[kernel.length][kernel[0].length]
                            [kernel[0][0].length][kernel[0][0].length];
                    for (int h = 0; h < kernel.length; h++) {

                        for (int j = 0; j < prevD[h].length; j++) {
                            for (int k = 0; k < prevD[h].length; k++) {

                                for (int z = 0; z < extended.length; z++) {
                                    for (int x = s * j; x < s * j + this.k; x++) {
                                        for (int y = s * k; y < s * k + this.k; y++) {
                                            kernelDer[h][z][x - s * j][y - s * k] +=
                                                    extended[z][x][y] * prevD[h][j][k];
                                            kernelNorms[h][z][x - s * j][y - s * k] +=
                                                    Math.pow(kernelDer[h][z][x - s * j][y - s * k], 2);
                                        }
                                    }
                                }

                            }
                        }

                    }
                }
            }
        }
    }

    public static double[][][] extend3D(double[][][] base, int p, Type type) {
        double[][][] res = new double[base.length][][];
        for (int i = 0; i < base.length; i++) {
            res[i] = extend2D(base[i], p, type);
        }
        return res;
    }

    public static double[][] extend2D(double[][] base, int p, Type type) {
        int n = base.length;
        double[][] extended = new double[n + 2 * p][n + 2 * p];
        for (int i = -p; i < n + p; i++) {
            for (int j = -p; j < n + p; j++) {
                Pair<Integer, Integer> newCoords;
                if (type == Type.CNVC) {
                    newCoords = getCyclicCoords(i, j, n);
                } else if (type == Type.CNVE) {
                    newCoords = getExtendedCoords(i, j, n);
                } else {
                    newCoords = getMirroredCoords(i, j, n);
                }
                extended[i + p][j + p] = base[newCoords.getKey()][newCoords.getValue()];
            }
        }
        return extended;
    }

    public static double[][][] fold3D(double[][][] base, int p, Type type) {
        double[][][] res = new double[base.length][][];
        for (int i = 0; i < base.length; i++) {
            res[i] = fold2D(base[i], p, type);
        }
        return res;
    }

    public static double[][] fold2D(double[][] base, int p, Type type) {
        int n = base.length;
        double[][] folded = new double[n - 2 * p][n - 2 * p];
        for (int i = -p; i < n - p; i++) {
            for (int j = -p; j < n - p; j++) {
                Pair<Integer, Integer> newCoords;
                if (type == Type.CNVC) {
                    newCoords = getCyclicCoords(i, j, n - 2 * p);
                } else if (type == Type.CNVE) {
                    newCoords = getExtendedCoords(i, j, n - 2 * p);
                } else {
                    newCoords = getMirroredCoords(i, j, n - 2 * p);
                }
                folded[newCoords.getKey()][newCoords.getValue()] += base[i + p][j + p];
            }
        }
        return folded;
    }

    public static Pair<Integer, Integer> getMirroredCoords(int j, int k, int n) {
        while (j < 0 || j >= n) {
            if (j < 0) j = -j;
            else j -= 2 * (j - n + 1);
        }
        while (k < 0 || k >= n) {
            if (k < 0) k = -k;
            else k -= 2 * (k - n + 1);
        }
        return new Pair<>(j, k);
    }

    public static Pair<Integer, Integer> getExtendedCoords(int j, int k, int n) {
        while (j < 0 || j >= n) {
            if (j < 0) j = 0;
            else j = n - 1;
        }
        while (k < 0 || k >= n) {
            if (k < 0) k = 0;
            else k = n - 1;
        }
        return new Pair<>(j, k);
    }

    public static Pair<Integer, Integer> getCyclicCoords(int j, int k, int n) {
        while (j < 0 || j >= n) {
            if (j < 0) j = (j + 50 * n) % n;
            else j = j % n;
        }
        while (k < 0 || k >= n) {
            if (k < 0) k = (k + 50 * n) % n;
            else k = k % n;
        }
        return new Pair<>(j, k);
    }

    public static String matrixToString4D(double[][][][] matrix) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) sb.append(matrixToString3D(matrix[i]));
        return sb.toString();
    }

    public static String matrixToString3D(double[][][] matrix) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                for (int k = 0; k < matrix[0][0].length; k++) {
                    sb.append(matrix[i][j][k]).append(" ");
                }
            }
        }
        return sb.toString();
    }

    public static int classify(Layer[] layers, double[][][] matrix) {
        for (int layer = 0; layer < layers.length; layer++) {
            matrix = layers[layer].process(matrix);
        }
        int maxCl = 0;
        for (int i = 0; i < 10; i++) {
            if (matrix[maxCl][0][0] < matrix[i][0][0]) maxCl = i;
        }
        return maxCl;
    }

    public static void softArgMax(double[] array) {
        double sum = 0;
        double max = 0;
        for (int i = 0; i < array.length; i++) if (array[i] > max) max = array[i];
        for (double d : array) sum += Math.exp(d - max);
        for (int i = 0; i < array.length; i++) array[i] = Math.exp(array[i] - max) / sum;
    }

    public static double[][][] shortToDouble(short[][][] arr) {
        double[][][] res = new double[arr.length][arr[0].length][arr[0][0].length];
        for (int x = 0; x < arr.length; x++) {
            for (int y = 0; y < arr[0].length; y++) {
                for (int z = 0; z < arr[0][0].length; z++) {
                    res[x][y][z] = arr[x][y][z];
                }
            }
        }
        return res;
    }

    public static void train(Layer[] layers, short[][][][] input, int[] classes,
                             short[][][][] testSet, int[] testClasses) throws IOException {
        BufferedWriter log = new BufferedWriter(new FileWriter("log.txt"));
        Random R = new Random();
        XYSeriesCollection ds = new XYSeriesCollection();
        XYSeries series = new XYSeries("");
        double learningRate = 0.001;
        double eps = 1e-10;
        for (int epoch = 0; epoch < 100000; epoch++) {
            double[] trueClassesFreq = new double[10];
            int index = R.nextInt(input.length);
            trueClassesFreq[classes[index]]++;
            for (int steps = 0; steps < 20; steps++) {
                double[][][] matrix = shortToDouble(input[index]);
                for (Layer layer : layers) {
                    matrix = layer.process(matrix);
                }
                double[] resultClassesFreq = new double[10];
                for (int i = 0; i < 10; i++) resultClassesFreq[i] = matrix[i][0][0];
                softArgMax(resultClassesFreq);
                double[][][] derivative = new double[10][1][1];
                for (int cl = 0; cl < 10; cl++) derivative[cl][0][0] = resultClassesFreq[cl] - trueClassesFreq[cl];
                for (int layer = layers.length - 1; layer > -1; layer--) {
                    layers[layer].calcDerivative(derivative, matrix);
                    derivative = layers[layer].derivative;
                    matrix = layers[layer].matrix;
                    if (layers[layer].type == Type.CNVM) {

                        for (int h = 0; h < layers[layer].kernel.length; h++) {
                            for (int z = 0; z < layers[layer].kernel[0].length; z++) {
                                for (int x = 0; x < layers[layer].kernel[0][0].length; x++) {
                                    for (int y = 0; y < layers[layer].kernel[0][0][0].length; y++) {
                                        layers[layer].kernel[h][z][x][y] -=
                                                (learningRate / Math.sqrt(eps + layers[layer].kernelNorms[h][z][x][y]))
                                                        * layers[layer].kernelDer[h][z][x][y];
                                    }
                                }
                            }
                        }

                    }
                }

            }
            int wrongCount = 0;
            if (epoch % 500 == 0) {
                for (int i = 0; i < testSet.length; i++) {
                    int cl = classify(layers, shortToDouble(testSet[i]));
                    if (cl != testClasses[i]) wrongCount++;
                }
                series.add(epoch, (double) wrongCount / testSet.length);
                log.write("Epoch " + epoch + ": error rate = " + (double) wrongCount / testSet.length + "\n");
                log.flush();
            }
        }
        ds.addSeries(series);
        Utils.writeChartForDS("Error rate to epoch-Fashion", ds, "Epoch", "Error rate");
    }

    public static void fillKernel(double[][][][] kernel) {
        Random R = new Random();
        for (int h = 0; h < kernel.length; h++) {
            for (int z = 0; z < kernel[0].length; z++) {
                for (int x = 0; x < kernel[0][0].length; x++) {
                    for (int y = 0; y < kernel[0][0][0].length; y++) {
                        kernel[h][z][x][y] = (R.nextDouble() / 10) * (R.nextDouble() > 0.5 ? 1 : -1);
                    }
                }
            }
        }
    }

    public static void main(String[] args) throws IOException {
        List<String[]> values = new CSVReader(new FileReader("src/main/resources/fashion-mnist_train.csv")).readAll();
        values.remove(0);  // header
        int samples = values.size();
        short[][][][] trainSet = new short[samples][1][28][28];
        int[] trainClasses = new int[samples];
        for (int i = 0; i < samples; i++) {
            String[] l = values.get(i);
            trainClasses[i] = Integer.valueOf(l[0]);
            int index = 1;
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    trainSet[i][0][x][y] = Short.valueOf(l[index++]);
                }
            }
        }
        List<String[]> valuesTest = new CSVReader(new FileReader("src/main/resources/fashion-mnist_test.csv")).readAll();
        valuesTest.remove(0);  // header
        int samplesTest = valuesTest.size();
        short[][][][] testSet = new short[samplesTest][1][28][28];
        int[] testClasses = new int[samplesTest];
        for (int i = 0; i < samplesTest; i++) {
            String[] l = valuesTest.get(i);
            testClasses[i] = Integer.valueOf(l[0]);
            int index = 1;
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    testSet[i][0][x][y] = Short.valueOf(l[index++]);
                }
            }
        }
        double[][][][] kernelFirst = new double[5][1][3][3];
        fillKernel(kernelFirst);
        double[][][][] kernelSecond = new double[10][5][3][3];
        fillKernel(kernelSecond);
        double[][][][] kernelThird = new double[15][10][3][3];
        fillKernel(kernelThird);
        double[][][][] kernelLast = new double[10][15][7][7];
        fillKernel(kernelLast);
        Layer[] layers = new Layer[6];
        layers[0] = new Layer(kernelFirst, 5, 3, 1, 1, "CNVM");
        layers[1] = new Layer(2);
        layers[2] = new Layer(kernelSecond, 10, 3, 1, 1, "CNVM");
        layers[3] = new Layer(2);
        layers[4] = new Layer(kernelThird, 15, 3, 1, 1, "CNVM");
        layers[5] = new Layer(kernelLast, 10, 7, 1, 0, "CNVM");
        train(layers, trainSet, trainClasses, testSet, testClasses);
    }

}