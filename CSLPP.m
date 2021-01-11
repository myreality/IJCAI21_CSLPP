function [rddata] = CSLPP(X,Y,wd,k,m,q,lemmda)
countloss=0;
Loss=[];
[data_row,data_col] =size(X);
cluster_n=length(unique(Y)); 
% Make sure data is zero mean
mapping.mean = mean(X, 1);
X = bsxfun(@minus, X, mapping.mean);
St=X'*X;
% Perform eigendecomposition of C
St(isnan(St)) = 0;
St(isinf(St)) = 0;
U=rand(cluster_n,data_row);
[sp,ps]=sort(U,1);
U(ps>k)=0;
col_sum=sum(U);
U=U./col_sum;
mf = U.^m;
S=rand(data_row,data_row);
Scol_sum=sum(S);
S=S./Scol_sum;
% W=eye(data_col,wd);
maxgen =50;
for i = 1:maxgen
    %step 1 fixing w get smf and center
    center = mf*X./(repmat(sum(mf,2),1,data_col));
    dn=diag(sum(mf,1));
    dc=diag(sum(mf,2));
    SS=S.^q;
    D=diag(sum(SS,2));
    L=D-0.5*(SS+SS');
    M=(X'*dn*X-2*X'*mf'*center+center'*dc*center+2*lemmda*X'*L*X);
    M=(M+M')/2;
    M(find(isnan(M)==1))= 0;
    G=chol(St,'lower');
    MM=inv(G)*M*(inv(G))';
    [W,B] = eig(MM);
    B(isnan(B)) = 0;
   [B, ind] = sort(diag(B));
    W = (inv(G))'*W;
    W = W(:,ind(1:wd)); 
    ldata =real(X*W);
    lcenter =real(center*W);
    ldist =L2_distance_subfun(ldata',lcenter');
    distx = L2_distance_subfun((X*W)',(X*W)');
    Lossvalue=real(lemmda*sum(sum((S.^q).*distx))+sum(sum(ldist'.*mf)));
    if isnan(Lossvalue)
        break;
    end
    Loss=[Loss,Lossvalue];
%        if countloss>1 
%          if Loss(i-1)-Loss(i)<10^-5
%                    break;
%          end
%      end
  
    %[tmpp, ind] = sort(ldist,2);
    ldist(ldist ~= 0) = ldist(ldist ~= 0).^(-1/(m-1))';
    tmp=ldist';
    U=tmp./sum(tmp);
    mf = U.^m;
  
    %mydis=L2_distance_1((X*W)',(X*W)');
    [tmpp2, ind2] = sort(distx,2);
    for j=1:size(distx, 1)
        distx(j, ind2(j,(1 + k):data_row)) = 0;
        distx(j, j) = 0;
    end
    distx(distx ~= 0) = distx(distx ~= 0).^(-1/(q-1))';
    tmp2=distx';
    U=tmp2./sum(tmp2);
    S = U.^q;
   
    S=(S+S')/2;
    countloss=countloss+1;
end
 rddata =real(X*W);
end


