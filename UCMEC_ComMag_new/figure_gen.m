% Performance Evaluation

Rate_user_num = zeros(6,6);
Rate_ap_num = zeros(6,6);
Rate_max_power = zeros(6,6);
Rate_cluster_size = zeros(6,6);

Delay_user_num = zeros(6,6);
Delay_ap_num = zeros(6,6);
Delay_max_power = zeros(6,6);
Delay_cluster_size = zeros(6,6);


% Figure 3: User Num VS Uplink Rate
User_index = [5,10,15,20,25,30];

plot(User_index,Rate_user_num(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(User_index,Rate_user_num(2,:),"-+",'Markersize',7,'linewidth',2);
plot(User_index,Rate_user_num(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(User_index,Rate_user_num(4,:),"-x",'Markersize',7,'linewidth',2);
plot(User_index,Rate_user_num(5,:),"-*",'Markersize',7,'linewidth',2);
plot(User_index,Rate_user_num(6,:),"-p",'Markersize',7,'linewidth',2);

grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Number of Users'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',15);


% Figure 4: AP Num VS Uplink Rate
AP_Num_index = [10,20,30,40,50,60];

plot(AP_Num_index,Rate_ap_num(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(AP_Num_index,Rate_ap_num(2,:),"-+",'Markersize',7,'linewidth',2);
plot(AP_Num_index,Rate_ap_num(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(AP_Num_index,Rate_ap_num(4,:),"-x",'Markersize',7,'linewidth',2);
plot(AP_Num_index,Rate_ap_num(5,:),"-*",'Markersize',7,'linewidth',2);
plot(AP_Num_index,Rate_ap_num(6,:),"-p",'Markersize',7,'linewidth',2);

grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Number of APs'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',15);


% Figure 5: Maximum Power VS Uplink Rate
Max_power_index = [0.05,0.1,0.15,0.2,0.25,0.3];

plot(Max_power_index,Rate_max_power(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(Max_power_index,Rate_max_power(2,:),"-+",'Markersize',7,'linewidth',2);
plot(Max_power_index,Rate_max_power(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(Max_power_index,Rate_max_power(4,:),"-x",'Markersize',7,'linewidth',2);
plot(Max_power_index,Rate_max_power(5,:),"-*",'Markersize',7,'linewidth',2);
plot(Max_power_index,Rate_max_power(6,:),"-p",'Markersize',7,'linewidth',2);


grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Maximum Transmit Power (W)'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',15);

% Figure 6: AP Cluster Size VS Uplink Rate
AP_cluster_index = [1,2,3,4,5,6];

plot(AP_cluster_index,Rate_cluster_size(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(AP_cluster_index,Rate_cluster_size(2,:),"-+",'Markersize',7,'linewidth',2);
plot(AP_cluster_index,Rate_cluster_size(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(AP_cluster_index,Rate_cluster_size(4,:),"-x",'Markersize',7,'linewidth',2);
plot(AP_cluster_index,Rate_cluster_size(5,:),"-*",'Markersize',7,'linewidth',2);
plot(AP_cluster_index,Rate_cluster_size(6,:),"-p",'Markersize',7,'linewidth',2);

grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('AP Cluster Size'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',15);



% Figure 7: User Num VS Average Total Delay
User_index = [5,10,15,20,25,30];

plot(User_index,Delay_user_num(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(User_index,Delay_user_num(2,:),"-+",'Markersize',7,'linewidth',2);
plot(User_index,Delay_user_num(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(User_index,Delay_user_num(4,:),"-x",'Markersize',7,'linewidth',2);
plot(User_index,Delay_user_num(5,:),"-*",'Markersize',7,'linewidth',2);
plot(User_index,Delay_user_num(6,:),"-p",'Markersize',7,'linewidth',2);


grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Number of Users'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',15);


% Figure 8: AP Num VS Average Total Delay
AP_Num_index = [10,20,30,40,50,60];

plot(AP_Num_index,Delay_ap_num(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(AP_Num_index,Delay_ap_num(2,:),"-+",'Markersize',7,'linewidth',2);
plot(AP_Num_index,Delay_ap_num(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(AP_Num_index,Delay_ap_num(4,:),"-x",'Markersize',7,'linewidth',2);
plot(AP_Num_index,Delay_ap_num(5,:),"-*",'Markersize',7,'linewidth',2);
plot(AP_Num_index,Delay_ap_num(6,:),"-p",'Markersize',7,'linewidth',2);


grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Number of APs'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',15);


% Figure 9: Maximum Power VS Average Total Delay
Max_power_index = [0.05,0.1,0.15,0.2,0.25,0.3];

plot(Max_power_index,Delay_max_power(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(Max_power_index,Delay_max_power(2,:),"-+",'Markersize',7,'linewidth',2);
plot(Max_power_index,Delay_max_power(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(Max_power_index,Delay_max_power(4,:),"-x",'Markersize',7,'linewidth',2);
plot(Max_power_index,Delay_max_power(5,:),"-*",'Markersize',7,'linewidth',2);
plot(Max_power_index,Delay_max_power(6,:),"-p",'Markersize',7,'linewidth',2);


grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Maximum Transmit Power (W)'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',15);

% Figure 10: AP Cluster Size VS Average Total Delay
AP_cluster_index = [1,2,3,4,5,6];

plot(AP_cluster_index,Delay_cluster_size(1,:),"-o",'Markersize',7,'linewidth',2); hold on;
plot(AP_cluster_index,Delay_cluster_size(2,:),"-+",'Markersize',7,'linewidth',2);
plot(AP_cluster_index,Delay_cluster_size(3,:),"-d",'Markersize',7,'linewidth',2); 
plot(AP_cluster_index,Delay_cluster_size(4,:),"-x",'Markersize',7,'linewidth',2);
plot(AP_cluster_index,Delay_cluster_size(5,:),"-*",'Markersize',7,'linewidth',2);
plot(AP_cluster_index,Delay_cluster_size(6,:),"-p",'Markersize',7,'linewidth',2);

grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('AP Cluster Size'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',15);
