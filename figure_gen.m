% Performance Evaluation

Rate_user_num = zeros(6,6);
Rate_ap_num = zeros(6,6);
Rate_max_power = zeros(6,6);
Rate_cluster_size = zeros(6,6);

Delay_user_num = zeros(6,6);
Delay_ap_num = zeros(6,6);
Delay_max_power = zeros(6,6);
Delay_cluster_size = zeros(6,6);

% Figure 1: Convergence Performance
episode_index = 1:100;0.

load('reward_proposed');
load('reward_proposed');
load('reward_proposed');
load('reward_proposed');
load('reward_proposed');
load('reward_proposed');


plot(episode_index,reward_proposed,'linewidth',2); hold on;
plot(episode_index,reward_cbo,"-+",'Markersize',4,'linewidth',2); 
plot(episode_index,reward_acra,"-d",'Markersize',4,'linewidth',2); 
plot(episode_index,reward_mpbo,"-x",'Markersize',4,'linewidth',2); 
plot(episode_index,reward_rpbo,"-*",'Markersize',4,'linewidth',2); 
plot(episode_index,reward_ddpg,"-p",'Markersize',4,'linewidth',2); 


grid on;
legend('Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG");
xlabel('Step Index'),ylabel('Reward');
set(gca,'FontName','Times New Roman','FontSize',15);



% Figure 2: Training Time
%If you want to adjust the pattern to 6 bar such as " applyhatch(gcf,'.-+/|x');",
%try to type this "applyhatch(gcf,'.-++/||xx');" instedly. 
%So you can avoid the duplicated pattern at least, even order problem is still not solved. 
% time_list = zeros(6,3);
X = [1,2,3,4,5,6];

GO = bar(X,time_list,1,'EdgeColor','k','LineWidth',1);

hatchfill2(GO(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(3),'single','HatchAngle',0,'HatchDensity',40,'HatchColor','k');

GO(1).FaceColor = [0.000, 0.447, 0.741];
GO(2).FaceColor = [0.850, 0.325, 0.098];
GO(3).FaceColor = [0.929, 0.694, 0.125];

% Draw the legend
legendData = {'N = 10','N = 20','N = 30'};
[legend_h, object_h, plot_h, text_str] = legendflex(GO, legendData, 'Padding', [2, 2, 10], 'FontSize', 15, 'Location', 'NorthEast');
% object_h(1) is the first bar's text
% object_h(2) is the second bar's text
% object_h(3) is the first bar's patch
% object_h(4) is the second bar's patch
%
% Set the two patches within the legend
hatchfill2(object_h(4), 'cross', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(5), 'single', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(6), 'single', 'HatchAngle', 0, 'HatchDensity', 40, 'HatchColor', 'k');

% Some extra formatting to make it pretty :)
set(gca, 'FontSize', 11);
set(gca, 'XMinorTick','on', 'XMinorGrid','on', 'YMinorTick','on', 'YMinorGrid','on');
% xlim([0.5, 2.5]);
ylim([0, 800]);
grid on;
% hTitle = title('Texture filled bar chart');
% hXLabel = xlabel('Samples');
hYLabel = ylabel('Training Time for Convergence');
barNames = {'Proposed','CBO', "ACRA", "MPBO", "RPBO", "DDPG"};
xticklabels(barNames);
set(gca,'FontName','Times New Roman','FontSize',15);


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
