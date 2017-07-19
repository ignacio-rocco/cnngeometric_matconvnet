function sizestr = size2string(sizevec)


if length(sizevec)>=1
    sizestr=num2str(sizevec(1));
    if length(sizevec)>=2
        for i=2:length(sizevec)
            sizestr=[sizestr 'x' num2str(sizevec(i))];
        end
    end
else
    sizestr='';
end


    