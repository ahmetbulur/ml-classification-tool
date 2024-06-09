using java.util;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using weka.core;



namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {

        static object[] inputs; 

        //DATASETS
        static weka.core.Instances insts;
        static weka.core.Instances insts2;

        
        static Boolean isNumeric = false;
        static Boolean isNominal = false;

       
        const int percentSplit = 66; 

        
        static weka.classifiers.Classifier cl = null; //FOR BEST MODEL

        //MODELS FOR OUR ALGORITHMS
        static weka.classifiers.Classifier SMOcl = null;
        static weka.classifiers.Classifier NaiveBayescl = null;
        static weka.classifiers.Classifier RandomForestcl = null;
        static weka.classifiers.Classifier RandomTreecl = null;
        static weka.classifiers.Classifier MultiLayerPerceptroncl = null;
        static weka.classifiers.Classifier J48cl = null;
        static weka.classifiers.Classifier _1BKcl = null;
        static weka.classifiers.Classifier _3BKcl = null;
        static weka.classifiers.Classifier _5BKcl = null;
        static weka.classifiers.Classifier LogisticRegressioncl = null;


        

        //MODEL METHODS
        public static double SMOclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                SMOcl = new weka.classifiers.functions.SMO();


                
                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);

                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                SMOcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = SMOcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

            public static double NaiveBayesclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                NaiveBayescl = new weka.classifiers.bayes.NaiveBayes();

                
                weka.filters.Filter myDisc = new weka.filters.unsupervised.attribute.Discretize();
                myDisc.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDisc);


                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                NaiveBayescl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = NaiveBayescl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        //D-TREE ALGORITHM
        public static double RandomForestclassifyTest(weka.core.Instances insts)
        {
            try
            {

                insts.setClassIndex(insts.numAttributes() - 1);

                RandomForestcl = new weka.classifiers.trees.RandomForest();

                
                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                RandomForestcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = RandomForestcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        //D-TREE ALG.2
        public static double RandomTreeclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                RandomTreecl = new weka.classifiers.trees.RandomTree();

                
                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                RandomTreecl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = RandomTreecl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }
        
        //D-TREE ALG.3
        public static double J48classifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                J48cl = new weka.classifiers.trees.J48();

               
                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);


                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                J48cl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = J48cl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        
        public static double MultiLayerPerceptronclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                MultiLayerPerceptroncl = new weka.classifiers.functions.MultilayerPerceptron();

                
                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);

                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                MultiLayerPerceptroncl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = MultiLayerPerceptroncl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        //K-Nearest Neighbour 
        public static double _1BKclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                _1BKcl = new weka.classifiers.lazy.IBk(1);

                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);

                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _1BKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _1BKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        //K-Nearest Neighbour 2
        public static double _3BKclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                _3BKcl = new weka.classifiers.lazy.IBk(3);

                
                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);

                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _3BKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _3BKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        //K-Nearest Neighbour 3
        public static double _5BKclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                _5BKcl = new weka.classifiers.lazy.IBk(5);

                
                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);

                 weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                 myNormalized.setInputFormat(insts);
                 insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _5BKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _5BKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        //Logistic Regression
        public static double LogisticRegressionclassifyTest(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                LogisticRegressioncl = new weka.classifiers.functions.Logistic();


                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);

                weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
                myNormalized.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalized);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                LogisticRegressioncl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = LogisticRegressioncl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static void BestAlgorithmFinding(Panel panel2, TextBox textbox1, Label label1)
        {

            insts = new weka.core.Instances(new java.io.FileReader(textbox1.Text));
            insts2 = new weka.core.Instances(new java.io.FileReader(textbox1.Text)); 

            double max_value = 0;

            double J48_rate = max_value = J48classifyTest(insts);
            insts = insts2;
            cl = J48cl;
            isNominal = false;
            isNumeric = false;
            
            double RandomForest_rate = RandomForestclassifyTest(insts);
            insts = insts2;

            if (RandomForest_rate > max_value)
            {
                cl = RandomForestcl;
                max_value = RandomForest_rate;
            }

            double RandomTree_rate = RandomTreeclassifyTest(insts);
            insts = insts2;

            if (RandomTree_rate > max_value)
            {
                cl = RandomTreecl;
                max_value = RandomTree_rate;
            }

            double MultiLayerPerceptron_rate = MultiLayerPerceptronclassifyTest(insts);
            insts = insts2;

            if (MultiLayerPerceptron_rate > max_value)
            {
                cl = MultiLayerPerceptroncl;
                max_value = MultiLayerPerceptron_rate;
                isNominal = false;
                isNumeric = true;
            }

            double SMO_rate = SMOclassifyTest(insts);
            insts = insts2;

            if (SMO_rate > max_value)
            {
                cl = SMOcl;
                max_value = SMO_rate;
                isNominal = false;
                isNumeric = true;
            }

            double NaiveBayes_rate = NaiveBayesclassifyTest(insts);
            insts = insts2;

            if (NaiveBayes_rate > max_value)
            {
                cl = NaiveBayescl;
                max_value = NaiveBayes_rate;
                isNominal = true;
                isNumeric = false;
            }

            double _1BK_rate = _1BKclassifyTest(insts);
            insts = insts2;

            if (_1BK_rate > max_value)
            {
                cl = _1BKcl;
                max_value = _1BK_rate;
                isNominal = false;
                isNumeric = true;
            }

            double _3BK_rate = _3BKclassifyTest(insts);
            insts = insts2;

            if (_3BK_rate > max_value)
            {
                cl = _3BKcl;
                max_value = _3BK_rate;
                isNominal = false;
                isNumeric = true;
            }

            double _5BK_rate = _5BKclassifyTest(insts);
            insts = insts2;

            if (_5BK_rate > max_value)
            {
                cl = _5BKcl;
                max_value = _5BK_rate;
                isNominal = false;
                isNumeric = true;
            }

            double LogisticRegression_rate = LogisticRegressionclassifyTest(insts);
            insts = insts2;

            if (LogisticRegression_rate > max_value)
            {
                cl = LogisticRegressioncl;
                max_value = LogisticRegression_rate;
                isNominal = false;
                isNumeric = true;
            }

            insts = insts2;


            label1.Text = cl.GetType().Name + " is the most successful algorithm for this dataset (%" + max_value + ")";
            label1.Visible = true;
            
            inputs = new object[insts.numAttributes() - 1];

            try
            {
                int pointX = 30;
                int pointY = 40;

                for (int i = 0; i < inputs.Length; i++)
                {
                   
                    Label l = new Label();
                    l.Text = insts.attribute(i).name();
                    l.Location = new Point(pointX, pointY);
                    panel2.Controls.Add(l);

                    if (insts.attribute(i).isNumeric())
                    {
                        
                        TextBox t = new TextBox();
                        t.Text = insts.instance(0).value(i).ToString(); 
                        t.Location = new Point(pointX + 100, pointY);
                        panel2.Controls.Add(t);
                        pointY += 20;

                        inputs[i] = t;
                    }
                    else
                    {
                        
                       
                        Enumeration vals = insts.attribute(i).enumerateValues();
                        object[] distVals = Collections.list(vals).toArray();

                        ComboBox c = new ComboBox();
                        c.Items.AddRange(distVals.Cast<string>().ToArray());
                        c.SelectedIndex = 0;
                        c.Location = new Point(pointX + 100, pointY);
                        panel2.Controls.Add(c);
                        pointY += 20;

                        inputs[i] = c;
                    }
                }
            }
            catch (Exception)
            {
                MessageBox.Show("Please select available dataset which has nominal target attribute");
            }
        } 


        public Form1()
        {

            InitializeComponent();
            
        }


        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void panel2_Paint(object sender, PaintEventArgs e)
        {

        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            
            weka.core.Instance predInst = new DenseInstance(insts.numAttributes());
            insts = insts2;
            predInst.setDataset(insts);
            
           
            for (int i = 0; i < inputs.Length; i++)
            {
                if (insts.attribute(i).isNumeric())
                    predInst.setValue(i, Convert.ToDouble(((TextBox)inputs[i]).Text));
                else
                    predInst.setValue(i, ((ComboBox)inputs[i]).SelectedIndex);
            }

            insts.add(predInst);

            if (isNumeric)
            {
                weka.filters.Filter myNB = new weka.filters.unsupervised.attribute.NominalToBinary();
                myNB.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNB);
            }
            else if (isNominal)
            {
                weka.filters.Filter myDisc = new weka.filters.unsupervised.attribute.Discretize();
                myDisc.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDisc);
            }

            weka.filters.Filter myNormalized = new weka.filters.unsupervised.attribute.Normalize();
            myNormalized.setInputFormat(insts);
            insts = weka.filters.Filter.useFilter(insts, myNormalized);


            
            double predClass = cl.classifyInstance(insts.instance(insts.numInstances()-1));


            Enumeration OutcomeValues = insts.classAttribute().enumerateValues();
            object[] classVals = Collections.list(OutcomeValues).toArray();

            this.label2.Text = "RESULT : " + classVals[(int)predClass].ToString();
            this.label2.Visible = true;

        }

        private void button2_Click(object sender, EventArgs e)
        {
           
            panel2.Controls.Clear();
            panel2.Visible = false;
            button1.Visible = false;
            label1.Visible = false;
            label2.Visible = false;

            OpenFileDialog fdlg = new OpenFileDialog();
            fdlg.Title = "C# Corner Open File Dialog";

            fdlg.InitialDirectory = @"C:\Users";
            fdlg.Filter = "All files (*.*)|*.*|All files (*.*)|*.*";
            fdlg.FilterIndex = 2;
            fdlg.RestoreDirectory = true;
            if (fdlg.ShowDialog() == DialogResult.OK)
            {
                textBox1.Text = fdlg.FileName;
            }

            BestAlgorithmFinding(panel2,textBox1,label1);
            panel2.Visible = true;
            button1.Visible = true;
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
