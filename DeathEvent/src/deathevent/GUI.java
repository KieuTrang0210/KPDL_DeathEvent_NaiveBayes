/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package deathevent;

import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

/**
 *
 * @author daothikieutrang
 */
public class GUI extends javax.swing.JFrame {

    /**
     * Creates new form GUI
     */
    NaiveBayesModel nb = new NaiveBayesModel();
    public GUI() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        buttonGroup1 = new javax.swing.ButtonGroup();
        btnPredict = new javax.swing.JButton();
        age = new javax.swing.JLabel();
        anaemia = new javax.swing.JLabel();
        creatinine_phosphokinase = new javax.swing.JLabel();
        diabetes = new javax.swing.JLabel();
        ejection_fraction = new javax.swing.JLabel();
        high_blood_pressure = new javax.swing.JLabel();
        platelets = new javax.swing.JLabel();
        serum_creatinine = new javax.swing.JLabel();
        serum_sodium = new javax.swing.JLabel();
        sex = new javax.swing.JLabel();
        smoking = new javax.swing.JLabel();
        time = new javax.swing.JLabel();
        txtAge = new javax.swing.JTextField();
        txtCreatininePhosphokinase = new javax.swing.JTextField();
        txtEjectionFraction = new javax.swing.JTextField();
        txtPlatelets = new javax.swing.JTextField();
        txtSerumCreatinine = new javax.swing.JTextField();
        txtSerumSodium = new javax.swing.JTextField();
        txtTime = new javax.swing.JTextField();
        cbxAnaemia = new javax.swing.JComboBox<>();
        cbxDiabetes = new javax.swing.JComboBox<>();
        cbxHighBloodPressure = new javax.swing.JComboBox<>();
        cbxSex = new javax.swing.JComboBox<>();
        cbxSmoking = new javax.swing.JComboBox<>();
        txtDeathEvent = new javax.swing.JLabel();
        txtErrAge = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        txtErrEjectionFraction = new javax.swing.JLabel();
        btnReset = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Predict Mortality Risk");

        btnPredict.setText("Predict");
        btnPredict.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnPredictActionPerformed(evt);
            }
        });

        age.setText("Age (tuổi):");

        anaemia.setText("Anaemia (thiếu máu):");

        creatinine_phosphokinase.setText("Creatinine Phosphokinase (enzym CPK - mcg/L):");

        diabetes.setText("Diabetes (tiểu đường):");

        ejection_fraction.setText("Ejection Fraction (phân suất tống máu %):");

        high_blood_pressure.setText("High Blood Pressure (cao huyết áp):");

        platelets.setText("Platelets (lượng tiểu cầu trong máu - kilo tiểu cầu / mL):");

        serum_creatinine.setText("Serum Creatinine (nồng độ creatinine huyết thanh - mg/dL):");

        serum_sodium.setText("Serum Sodium (nồng độ natri huyết thanh - mEq/L):");

        sex.setText("Sex:");

        smoking.setText("Smoking:");

        time.setText("Time (thời gian theo dõi - ngày):");

        txtAge.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtAgeActionPerformed(evt);
            }
        });

        txtPlatelets.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtPlateletsActionPerformed(evt);
            }
        });

        cbxAnaemia.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Yes", "No" }));

        cbxDiabetes.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Yes", "No" }));

        cbxHighBloodPressure.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Yes", "No" }));

        cbxSex.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Male", "Female" }));

        cbxSmoking.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Yes", "No" }));

        txtDeathEvent.setFont(new java.awt.Font("Segoe UI", 3, 18)); // NOI18N
        txtDeathEvent.setForeground(new java.awt.Color(51, 51, 255));

        txtErrAge.setFont(new java.awt.Font("Segoe UI", 2, 12)); // NOI18N
        txtErrAge.setForeground(new java.awt.Color(255, 0, 51));

        jLabel1.setFont(new java.awt.Font("Segoe UI", 2, 18)); // NOI18N
        jLabel1.setForeground(new java.awt.Color(51, 0, 204));
        jLabel1.setText("Predict:");

        txtErrEjectionFraction.setFont(new java.awt.Font("Segoe UI", 2, 12)); // NOI18N
        txtErrEjectionFraction.setForeground(new java.awt.Color(255, 0, 51));

        btnReset.setText("Reset");
        btnReset.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnResetActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(31, 31, 31)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                        .addGroup(layout.createSequentialGroup()
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                .addComponent(serum_sodium, javax.swing.GroupLayout.PREFERRED_SIZE, 285, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(sex, javax.swing.GroupLayout.PREFERRED_SIZE, 54, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(smoking, javax.swing.GroupLayout.PREFERRED_SIZE, 82, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(time, javax.swing.GroupLayout.PREFERRED_SIZE, 191, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGap(59, 59, 59)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                .addComponent(txtPlatelets, javax.swing.GroupLayout.DEFAULT_SIZE, 223, Short.MAX_VALUE)
                                .addComponent(txtSerumCreatinine)
                                .addComponent(txtSerumSodium)
                                .addComponent(txtTime)
                                .addComponent(cbxSmoking, 0, 223, Short.MAX_VALUE)
                                .addComponent(cbxSex, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                        .addGroup(layout.createSequentialGroup()
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                .addComponent(diabetes, javax.swing.GroupLayout.PREFERRED_SIZE, 143, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(ejection_fraction, javax.swing.GroupLayout.PREFERRED_SIZE, 234, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(high_blood_pressure, javax.swing.GroupLayout.PREFERRED_SIZE, 220, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(anaemia, javax.swing.GroupLayout.PREFERRED_SIZE, 134, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(age, javax.swing.GroupLayout.PREFERRED_SIZE, 104, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(creatinine_phosphokinase, javax.swing.GroupLayout.PREFERRED_SIZE, 260, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGap(84, 84, 84)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                .addComponent(txtErrEjectionFraction, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(txtCreatininePhosphokinase)
                                .addComponent(txtAge)
                                .addComponent(txtEjectionFraction)
                                .addGroup(layout.createSequentialGroup()
                                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                        .addComponent(txtErrAge, javax.swing.GroupLayout.PREFERRED_SIZE, 199, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(cbxHighBloodPressure, javax.swing.GroupLayout.PREFERRED_SIZE, 95, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(cbxDiabetes, javax.swing.GroupLayout.PREFERRED_SIZE, 95, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(cbxAnaemia, javax.swing.GroupLayout.PREFERRED_SIZE, 94, javax.swing.GroupLayout.PREFERRED_SIZE))
                                    .addGap(0, 24, Short.MAX_VALUE))))
                        .addGroup(layout.createSequentialGroup()
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                    .addComponent(platelets, javax.swing.GroupLayout.PREFERRED_SIZE, 308, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addGap(36, 36, 36))
                                .addGroup(layout.createSequentialGroup()
                                    .addComponent(serum_creatinine, javax.swing.GroupLayout.PREFERRED_SIZE, 326, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED))
                                .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                    .addComponent(btnPredict, javax.swing.GroupLayout.PREFERRED_SIZE, 94, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addGap(71, 71, 71)))
                            .addComponent(btnReset, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 65, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(txtDeathEvent, javax.swing.GroupLayout.PREFERRED_SIZE, 112, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap(75, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGap(11, 11, 11)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(age)
                    .addComponent(txtAge, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(1, 1, 1)
                .addComponent(txtErrAge)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cbxAnaemia, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(anaemia))
                .addGap(32, 32, 32)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(txtCreatininePhosphokinase, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(creatinine_phosphokinase))
                .addGap(29, 29, 29)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(diabetes)
                    .addComponent(cbxDiabetes, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(28, 28, 28)
                        .addComponent(ejection_fraction))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(25, 25, 25)
                        .addComponent(txtEjectionFraction, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(txtErrEjectionFraction)
                .addGap(27, 27, 27)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(high_blood_pressure)
                    .addComponent(cbxHighBloodPressure, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 28, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(platelets)
                    .addComponent(txtPlatelets, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(31, 31, 31)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(serum_creatinine)
                    .addComponent(txtSerumCreatinine, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(29, 29, 29)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(serum_sodium, javax.swing.GroupLayout.PREFERRED_SIZE, 29, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtSerumSodium, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(25, 25, 25)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(sex)
                    .addComponent(cbxSex, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(21, 21, 21)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cbxSmoking, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(smoking))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(time)
                    .addComponent(txtTime, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(txtDeathEvent, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addGap(22, 22, 22)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(btnReset, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnPredict, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(24, 24, 24))
        );

        txtAge.getAccessibleContext().setAccessibleName("txtAge");

        pack();
    }// </editor-fold>//GEN-END:initComponents

    @SuppressWarnings("empty-statement")
    private void btnPredictActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnPredictActionPerformed
        
        int age = Integer.parseInt(txtAge.getText());
        String age_label = "";
        if (age >= 0 && age <= 48) { age_label = "0_48";}
        else if (age > 48 && age <= 56) { age_label = "49_56";}
        else if (age > 56 && age <= 64) { age_label = "57_64";}
        else if (age > 64 && age <= 71) { age_label = "65_71";}
        else if (age > 71 && age <= 79) { age_label = "72_79";}
        else if (age > 79 && age <= 87) { age_label = "80_87";}
        else if (age > 87 && age <= 120) { age_label = "88_120";}
        else txtErrAge.setText("Invalid age");
        
        String anaemia = cbxAnaemia.getItemAt(cbxAnaemia.getSelectedIndex());
        String anaemia_label = "";
        if (anaemia.equals("Yes")) {
            anaemia_label = "1";
        } else {
             anaemia_label = "0";
        }
        
        int creatinine_phosphokinase = Integer.parseInt(txtCreatininePhosphokinase.getText());
        String creatinine_phosphokinase_label = "";
        if (creatinine_phosphokinase >= 0 && creatinine_phosphokinase <= 894) { creatinine_phosphokinase_label = "0_894";}
        else if (creatinine_phosphokinase > 894 && creatinine_phosphokinase <= 1745) { creatinine_phosphokinase_label = "895_1745";}
        else if (creatinine_phosphokinase > 1745 && creatinine_phosphokinase <= 2636) { creatinine_phosphokinase_label = "1746_2636";}
        else if (creatinine_phosphokinase > 2636 && creatinine_phosphokinase <= 3507) { creatinine_phosphokinase_label = "2637_3507";}
        else if (creatinine_phosphokinase > 3507 && creatinine_phosphokinase <= 4377) { creatinine_phosphokinase_label = "3508_4377";}
        else if (creatinine_phosphokinase > 4377 && creatinine_phosphokinase <= 5248) { creatinine_phosphokinase_label = "4378_5248";}
        else if (creatinine_phosphokinase > 5248 && creatinine_phosphokinase <= 6119) { creatinine_phosphokinase_label = "5249_6119";}
        else if (creatinine_phosphokinase > 6119 && creatinine_phosphokinase <= 6990) { creatinine_phosphokinase_label = "6120_6990";}
        else if (creatinine_phosphokinase > 6990) { creatinine_phosphokinase_label = "6991_max";}
 
        String diabetes = cbxDiabetes.getItemAt(cbxDiabetes.getSelectedIndex());
         String diabetes_label = "";
        if (anaemia.equals("Yes")) {
            diabetes_label = "1";
        } else {
             diabetes_label = "0";
        }
        
        int ejection_fraction = Integer.parseInt(txtEjectionFraction.getText());
        String ejection_fraction_label = "";
        if (ejection_fraction >= 0 && ejection_fraction <= 25) { ejection_fraction_label = "0_25";}
        else if (ejection_fraction > 25 && ejection_fraction <= 36) { ejection_fraction_label = "26_36";}
        else if (ejection_fraction > 36 && ejection_fraction <= 47) { ejection_fraction_label = "37_47";}
        else if (ejection_fraction > 47 && ejection_fraction <= 58) { ejection_fraction_label = "48_58";}
        else if (ejection_fraction > 58 && ejection_fraction <= 69) { ejection_fraction_label = "59_69";}
        else if (ejection_fraction > 69 && ejection_fraction <= 100) { ejection_fraction_label = "70_100";}
        else txtErrEjectionFraction.setText("Do not exceed 100%");
        
        String high_blood_pressure = cbxHighBloodPressure.getItemAt(cbxHighBloodPressure.getSelectedIndex());
          String high_blood_pressure_label = "";
        if (high_blood_pressure.equals("Yes")) {
            high_blood_pressure_label = "1";
        } else {
             high_blood_pressure_label = "0";
        }
        
        int platelets = Integer.parseInt(txtPlatelets.getText());
        String platelets_label = "";
        if (platelets >= 0 && platelets <= 116756) { platelets_label = "0_116756";}
        else if (platelets > 116756 && platelets <= 208411) { platelets_label = "116757_208411";}
        else if (platelets > 208411 && platelets <= 300067) { platelets_label = "208412_300067";}
        else if (platelets > 300067 && platelets <= 391722) { platelets_label = "300068_391722";}
        else if (platelets > 391722 && platelets <= 483378) { platelets_label = "391723_483378";}
        else if (platelets > 483378 && platelets <= 575033) { platelets_label = "483379_575033";}
        else if (platelets > 575033 && platelets <= 666689) { platelets_label = "575034_666689";}
        else if (platelets > 666689 && platelets <= 758344) { platelets_label = "666690_758344";}
        else if (platelets > 758344) { platelets_label = "758345_max";}

        Double serum_creatinine = Double.parseDouble(txtSerumCreatinine.getText());
        String serum_creatinine_label = "";
         if (serum_creatinine >= 0 && serum_creatinine <= 1.8) { serum_creatinine_label = "0_1.8";}
        else if (serum_creatinine > 1.8 && serum_creatinine <= 3.0) { serum_creatinine_label = "1.9_3.0";}
        else if (serum_creatinine > 3.0 && serum_creatinine <= 4.3) { serum_creatinine_label = "3.1_4.3";}
        else if (serum_creatinine > 4.3 && serum_creatinine <= 5.6) { serum_creatinine_label = "4.4_5.6";}
        else if (serum_creatinine > 5.6 && serum_creatinine <= 6.9) { serum_creatinine_label = "5.7_6.9";}
        else if (serum_creatinine > 6.9 && serum_creatinine <= 8.1) { serum_creatinine_label = "7.0_8.1";}
        else if (serum_creatinine > 8.1) { serum_creatinine_label = "8.2_max";}

        int serum_sodium = Integer.parseInt(txtSerumSodium.getText());
        String serum_sodium_label = "";
         if (serum_sodium >= 0 && serum_sodium <= 119) { serum_sodium_label = "0_119";}
        else if (serum_sodium > 119 && serum_sodium <= 125) { serum_sodium_label = "120_125";}
        else if (serum_sodium > 125 && serum_sodium <= 131) { serum_sodium_label = "126_131";}
        else if (serum_sodium > 131 && serum_sodium <= 136) { serum_sodium_label = "132_136";}
        else if (serum_sodium > 136 && serum_sodium <= 142) { serum_sodium_label = "137_142";}
        else if (serum_sodium > 142) { serum_sodium_label = "143_max";}
         
        String sex = cbxSex.getItemAt(cbxSex.getSelectedIndex());
        String sex_label = "";
        if (sex.equals("Male")) {
            sex_label = "1";
        } else {
             sex_label = "0";
        }
        
        String smoking = cbxSmoking.getItemAt(cbxSmoking.getSelectedIndex());
          String smoking_label = "";
        if (smoking.equals("Yes")) {
            smoking_label = "1";
        } else {
             smoking_label = "0";
        }
        
        int time = Integer.parseInt(txtTime.getText());
        String time_label = "";
        if (time >= 0 && time <= 35) { time_label = "0_35";}
        else if (time > 35 && time <= 66) { time_label = "36_66";}
        else if (time > 66 && time <= 98) { time_label = "67_98";}
        else if (time > 98 && time <= 129) { time_label = "99_129";}
        else if (time > 129 && time <= 160) { time_label = "130_160";}
        else if (time > 160 && time <= 191) { time_label = "161_191";}
        else if (time > 191 && time <= 223) { time_label = "192_223";}
        else if (time > 223 && time <= 254) { time_label = "224_254";}
        else if (time > 254) { time_label = "255_max";}
        
        try {
            String result = nb.predict(age_label, anaemia_label, creatinine_phosphokinase_label, diabetes_label, ejection_fraction_label, high_blood_pressure_label, platelets_label, serum_creatinine_label, serum_sodium_label, sex_label, smoking_label, time_label);
            txtDeathEvent.setText(result);
        } catch (Exception ex) {
            Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_btnPredictActionPerformed

    private void txtAgeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtAgeActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtAgeActionPerformed

    private void txtPlateletsActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtPlateletsActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtPlateletsActionPerformed

    private void btnResetActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnResetActionPerformed
      txtAge.setText("");
      txtCreatininePhosphokinase.setText("");
      txtDeathEvent.setText("");
      txtEjectionFraction.setText("");
      txtErrAge.setText("");
      txtErrEjectionFraction.setText("");
      txtPlatelets.setText("");
      txtSerumCreatinine.setText("");
      txtSerumSodium.setText("");
      txtTime.setText("");
      cbxAnaemia.setSelectedIndex(0);
      cbxDiabetes.setSelectedIndex(0);
      cbxHighBloodPressure.setSelectedIndex(0);
      cbxSex.setSelectedIndex(0);
      cbxSmoking.setSelectedIndex(0);
    }//GEN-LAST:event_btnResetActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new GUI().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel age;
    private javax.swing.JLabel anaemia;
    private javax.swing.JButton btnPredict;
    private javax.swing.JButton btnReset;
    private javax.swing.ButtonGroup buttonGroup1;
    private javax.swing.JComboBox<String> cbxAnaemia;
    private javax.swing.JComboBox<String> cbxDiabetes;
    private javax.swing.JComboBox<String> cbxHighBloodPressure;
    private javax.swing.JComboBox<String> cbxSex;
    private javax.swing.JComboBox<String> cbxSmoking;
    private javax.swing.JLabel creatinine_phosphokinase;
    private javax.swing.JLabel diabetes;
    private javax.swing.JLabel ejection_fraction;
    private javax.swing.JLabel high_blood_pressure;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel platelets;
    private javax.swing.JLabel serum_creatinine;
    private javax.swing.JLabel serum_sodium;
    private javax.swing.JLabel sex;
    private javax.swing.JLabel smoking;
    private javax.swing.JLabel time;
    private javax.swing.JTextField txtAge;
    private javax.swing.JTextField txtCreatininePhosphokinase;
    private javax.swing.JLabel txtDeathEvent;
    private javax.swing.JTextField txtEjectionFraction;
    private javax.swing.JLabel txtErrAge;
    private javax.swing.JLabel txtErrEjectionFraction;
    private javax.swing.JTextField txtPlatelets;
    private javax.swing.JTextField txtSerumCreatinine;
    private javax.swing.JTextField txtSerumSodium;
    private javax.swing.JTextField txtTime;
    // End of variables declaration//GEN-END:variables
}
