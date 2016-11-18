function [Maxi,Max1,Max2,sum] = NaiveBayse(x,Mn,Co)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
P=1;
Max1=0;
Max2=0;
sum=0;
Maxi=0;
Num_class=2;
Num_feat=87;
for i=1:Num_class
    P=1;
    for j=1:Num_feat
        P=P*(1./(sqrt(2*pi)*sqrt(Co(j,j,i))))*exp(-0.5*(((x(j)-Mn(j,i))^2)./Co(j,j,i)));
        sum=sum+P;
    end
    if(P>Max1)
        Max2=Max1;
        Max1=P;
        Maxi=i;
    else if(P>Max2)
            Max2=P;
        end
    end
end
