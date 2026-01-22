FROM node:20.16

ENV TZ=Asia/Seoul

WORKDIR /was

COPY ./package.json /was

RUN npm install

COPY . /was

CMD ["npm", "run", "start:prod" ]